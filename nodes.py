import json
import comfy
import torch
import math
import random
import folder_paths
import comfy.model_management as mm
import numpy as np
from pathlib import Path

# 获取当前脚本文件的目录
script_dir = Path(__file__).resolve().parent

from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from . import unet_2d_condition


def target_index(prompt, target):
    for index, i in enumerate(prompt.split()):
        if target == i:
            return torch.tensor([index + 1])


def convert_preview_image(images):
    # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
    images_tensors = []
    for img in images:
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(img)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


def binarize(attention):  # now attn has size(1, s1, 1)
    def maxmin_norm(x):
        norm = (x - torch.min(x, 1, keepdim=True)[0]) / (
                torch.max(x, 1, keepdim=True)[0] - torch.min(x, 1, keepdim=True)[0])
        return norm

    s = 10.
    attn_new = maxmin_norm((s * (maxmin_norm(attention) - 0.5)).sigmoid())
    return attn_new


def property_attn(attention_k, activation, device):  # activation has shape b, c, h, w
    # centroid
    shape = int(math.sqrt(attention_k.shape[1]))

    ls = torch.linspace(0, 1, steps=shape).to(device)  # activation.shape[-1]=innerD
    col = ls.repeat(shape, 1)
    row = col.t()  # transpose

    col = torch.stack([col], dim=0).view(1, -1)
    row = torch.stack([row], dim=0).view(1, -1)

    attn_col = (attention_k * col.unsqueeze(-1)).sum(1) / attention_k.sum(1)
    attn_row = (attention_k * row.unsqueeze(-1)).sum(1) / attention_k.sum(1)

    # shape
    attn_shape = binarize(attention_k)  # now attention_k has size (b, s1, 1)

    ### size
    attn_size = attn_shape.sum(1) / shape ** 2

    ### appearance
    actv = activation.view(activation.shape[0], activation.shape[1], -1)  # batch, channels, H*W
    appr = (actv.unsqueeze(2) * attn_shape.unsqueeze(1).transpose(2, 3)).sum(-1) / attn_shape.sum(1).unsqueeze(
        1)  # appr has shape b, c, 1
    # actv.unsqueeze(2)-> b, c, 1, HW
    # attn_shape.unsqueeze(1)-> b, 1, H*W, 1
    # .transpose(2,3)-> b, 1, 1, HW
    # *-> b, c, 1  HW
    # .sum->b, c, 1

    return attn_col, attn_row, attn_shape, attn_size, appr


class SG_HFModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["runwayml/stable-diffusion-v1-5"],
                    {"default": "runwayml/stable-diffusion-v1-5"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "TOKENIZER", "VAE",)
    RETURN_NAMES = ("model", "clip", "tokenizer", "vae",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "SelfGuidance"

    def load_checkpoint(self, model):
        device = mm.get_torch_device()
        with open(str(script_dir) + '/config.json') as f:
            unet_config = json.load(f)

        unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(model, subfolder="unet")
        tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(model, subfolder="vae")

        unet.to(device)
        text_encoder.to(device)
        vae.to(device)

        return (unet, text_encoder, tokenizer, vae,)


class EncodeText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokenizer": ("TOKENIZER",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a photo of a cute dog wearing a sweater and a baseball cap", }),
                "n_prompt": ("STRING", {"multiline": True,
                                        "default": "", }),
            }
        }

    RETURN_TYPES = ("EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "encode_text"
    CATEGORY = "SelfGuidance"

    def encode_text(self, tokenizer, clip, prompt, n_prompt):
        device = mm.get_torch_device()
        uncond_input = tokenizer(
            n_prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )

        uncond_embeddings = clip(uncond_input.input_ids.to(device))[0]
        input_ids = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

        cond_embeddings = clip(input_ids.input_ids.to(device))[0]
        embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        embedding_output = {
            "cond_embeddings": cond_embeddings,
            "embeddings": embeddings,
            "text": prompt
        }
        return (embedding_output,)


class NormalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "embedding": ("EMBEDDING",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, }),
            }
        }

    RETURN_TYPES = ("LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "normal_sampler"
    CATEGORY = "SelfGuidance"

    def normal_sampler(self, model, embedding, steps, seed):
        device = mm.get_torch_device()

        unet = model
        text_embedding = embedding["embeddings"]

        if seed is None:
            seed = random.randrange(2 ** 32 - 1)

        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (1, 4, 64, 64),
            generator=generator,
        ).to(device)

        noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                               beta_end=0.012,
                                               beta_schedule="scaled_linear",
                                               num_train_timesteps=1000)
        noise_scheduler.set_timesteps(steps)

        latents = latents * noise_scheduler.init_noise_sigma

        for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                noise_pred, \
                    attn_map_qk_up, \
                    attn_map_qk_mid, \
                    attn_map_qk_down, \
                    _, _, _ = \
                    unet(latent_model_input, t, encoder_hidden_states=text_embedding)

                noise_pred = noise_pred.sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                torch.cuda.empty_cache()
        return (latents,)


class DecodeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENTS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode_image"
    CATEGORY = "SelfGuidance"

    def decode_image(self, vae, latents):
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            output_images = convert_preview_image(pil_images)
        return (output_images,)


class AppControlSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "new_embedding": ("EMBEDDING",),
                "ori_embedding": ("EMBEDDING",),
                "target": ("STRING", {"multiline": False, "default": "sweater", }),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "ori_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, }),
                "new_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, }),
            }
        }

    RETURN_TYPES = ("LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "app_control"
    CATEGORY = "SelfGuidance"

    def app_control(self, model, new_embedding, ori_embedding, target, steps, ori_seed, new_seed):
        device = mm.get_torch_device()
        unet = model
        unet.enable_gradient_checkpointing()
        unet.train()

        ori_embeddings = ori_embedding["embeddings"]  # (2,77,768)
        ori_prompt = ori_embedding["text"]

        new_cond_embeddings = new_embedding["cond_embeddings"]
        new_embeddings = new_embedding["embeddings"]
        new_prompt = new_embedding["text"]

        # get position of target in each prompt
        position_ori = target_index(ori_prompt, target)
        position_new = target_index(new_prompt, target)

        # -----------------------------------------------------------------------------------------------------
        # seperate scheduler should be used for ori and new img
        noise_scheduler_ori = LMSDiscreteScheduler(beta_start=0.00085,
                                                   beta_end=0.012,
                                                   beta_schedule="scaled_linear",
                                                   num_train_timesteps=1000)
        noise_scheduler_ori.set_timesteps(steps)

        generator = torch.manual_seed(ori_seed)
        latents_ori = torch.randn(
            (1, 4, 64, 64),
            generator=generator,
        ).to(device)
        latents_ori = latents_ori * noise_scheduler_ori.init_noise_sigma

        # -----------------------------------------------------------------------------------
        noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                               beta_end=0.012,
                                               beta_schedule="scaled_linear",
                                               num_train_timesteps=1000)
        noise_scheduler.set_timesteps(steps)

        if new_seed is None:
            new_seed = random.randrange(2 ** 32 - 1)
        generator = torch.manual_seed(new_seed)
        latents_new = torch.randn(
            (1, 4, 64, 64),
            generator=generator,
        ).to(device)
        latents_new = latents_new * noise_scheduler.init_noise_sigma

        # -----------------------------------------------------------------------------------------
        # import pdb;pdb.set_trace()
        # update latent for new img
        for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
            # --------------- start recreate ori image
            with torch.no_grad():
                latent_model_input_ori = torch.cat([latents_ori] * 2)  # (2,4,64,64)
                latent_model_input_ori = noise_scheduler_ori.scale_model_input(latent_model_input_ori, t)  #
                noise_pred_ori, \
                    attn_map_qk_up_ori, \
                    attn_map_qk_mid_ori, \
                    attn_map_qk_down_ori, \
                    act_qkv_up_ori, \
                    act_qkv_mid_ori, \
                    act_qkv_down_ori = \
                    unet(latent_model_input_ori, t, encoder_hidden_states=ori_embeddings)
                # 三个阶段的attn,act,act是qkv之后的值，attn是qkv之前qk的值，act指的就是softmax激活函数
                # # (1,4,64,64)

                noise_pred_ori = noise_pred_ori.sample  # 2,4,64,64
                noise_pred_uncond_ori, noise_pred_text_ori = noise_pred_ori.chunk(2)
                noise_pred_ori = noise_pred_uncond_ori + 7.5 * (noise_pred_text_ori - noise_pred_uncond_ori)

                latents_ori = noise_scheduler_ori.step(noise_pred_ori, t, latents_ori).prev_sample  # (1,4,64,64)

            torch.cuda.empty_cache()
            # ------------------ end recreate ori image

            import pdb;
            pdb.set_trace()
            loss = torch.tensor(10000)
            iter = 0
            # latents_new = latents_new.clone().detach().requires_grad_(True)
            # --------------------- update latent for new img
            while loss.item() > 0.1 and ((index < 11 and iter < 5) or (
                    index % 5 == 0 and index < 25 and iter < 5)):  # you can change the threshold
                list_score = []
                # temperary forward of new img
                latents_new = latents_new.requires_grad_(True)
                # latents_new = latents_new.clone().detach().requires_grad_(True)

                latent_model_input = latents_new  # tmp
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                noise_pred, \
                    attn_map_qk_up, \
                    attn_map_qk_mid, \
                    attn_map_qk_down, \
                    act_qkv_up, \
                    act_qkv_mid, \
                    act_qkv_down = \
                    unet(latent_model_input, t, encoder_hidden_states=new_cond_embeddings)

                # obtain list of attention layers
                list_attn_ori = []
                list_act_ori = []

                for block1, block2 in zip(attn_map_qk_down_ori, act_qkv_down_ori):  # down的attn和激活函数之后map
                    for trans1, trans2 in zip(block1, block2):
                        list_attn_ori.append(trans1.chunk(2)[1])
                        list_act_ori.append(trans2.chunk(2)[1])

                for trans1, trans2 in zip(attn_map_qk_mid_ori, act_qkv_mid_ori):  # mid
                    list_attn_ori.append(trans1.chunk(2)[1])
                    list_act_ori.append(trans2.chunk(2)[1])

                for block1, block2 in zip(attn_map_qk_up_ori, act_qkv_up_ori):  # up
                    for trans1, trans2 in zip(block1, block2):
                        list_attn_ori.append(trans1.chunk(2)[1])
                        list_act_ori.append(trans2.chunk(2)[1])

                # obtain list_attn
                list_attn = []
                list_act = []

                for block1, block2 in zip(attn_map_qk_down, act_qkv_down):  #
                    for trans1, trans2 in zip(block1, block2):
                        list_attn.append(trans1)
                        list_act.append(trans2)

                for trans1, trans2 in zip(attn_map_qk_mid, act_qkv_mid):
                    list_attn.append(trans1)
                    list_act.append(trans2)

                for block1, block2 in zip(attn_map_qk_up, act_qkv_up):
                    for trans1, trans2 in zip(block1, block2):
                        list_attn.append(trans1)
                        list_act.append(trans2)

                # now list_attn contain attn-map elements
                w1 = 10.  # for weight of appr (if there are other prop)
                v = 10.  # for adjusting grad size

                # -------------------- extract object's attn slice of each map
                for idx, (attn, act) in enumerate(zip(list_attn, list_act)):
                    attn_ori = list_attn_ori[idx]
                    act_ori = list_act_ori[idx]

                    attention_k1 = attn[:, :, position_new].mean(-1, keepdim=True)  # sweater
                    attention_k1_ori = attn_ori[:, :, position_ori].mean(-1, keepdim=True)

                    # score_centroid = 2.0 * (torch.abs(aa_col -0.9).mean(-1) + torch.abs(aa_row - 0.2).mean(-1)) / 2.
                    # score_shape=10*torch.abs(aa_shape-aa_shape_org).mean(-2).mean(-1)

                    if idx not in [2, 4, 5, 6, 7, 8]:
                        # if idx not in [50]:
                        attn_col1, \
                            attn_row1, \
                            attn_shape1, \
                            attn_size1, \
                            attn_appr1 = property_attn(attention_k1, act, device=device)  # extract prop for k1

                        attn_col_ori, \
                            attn_row_ori, \
                            attn_shape_ori, \
                            attn_size_ori, \
                            attn_appr_ori = property_attn(attention_k1_ori, act_ori, device=device)

                        score_appr = w1 * torch.abs(attn_appr1 - attn_appr_ori).mean(-2).mean(-1)
                        list_score.append(score_appr)  # score in all maps
                # ------------------- end extraction object's feature

                # update latent by gradient decent
                import pdb;
                pdb.set_trace()
                loss = torch.mean(torch.stack(list_score, dim=0), dim=0)[0]
                # RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents_new], allow_unused=True)[0]
                # grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents_new])[0]

                # 获取模型的参数名称与参数值的映射
                model_state_dict = unet.state_dict()

                # 打印梯度为None的张量对应的参数名称
                for i, grad in enumerate(grad_cond):
                    if grad is None:
                        param_name = list(model_state_dict.keys())[i]
                        print(f"Gradient for parameter '{param_name}' is None")

                latents_new = latents_new - v * grad_cond * noise_scheduler.sigmas[
                    index]  # update new_latent according to the guidance
                torch.cuda.empty_cache()
                iter += 1

                print(f'iteration {iter}, loss={loss.item()}')
            # ---------------------- end update latent for new img

            with torch.no_grad():  # for new img
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                noise_pred, \
                    attn_map_qkv_up, \
                    attn_map_qkv_mid, \
                    attn_map_qkv_down, \
                    _, _, _ = \
                    unet(latent_model_input, t, encoder_hidden_states=new_embeddings)

                noise_pred = noise_pred.sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                torch.cuda.empty_cache()

        return (latents,)


NODE_CLASS_MAPPINGS = {
    "SG_HFModelLoader": SG_HFModelLoader,
    "EncodeText": EncodeText,
    "NormalSampler": NormalSampler,
    "DecodeImage": DecodeImage,
    "AppControlSampler": AppControlSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SG_HFModelLoader": "SG HFModelLoader",
    "EncodeText": "Encode Text",
    "NormalSampler": "Normal Sampler",
    "DecodeImage": "Decode Image",
    "AppControlSampler": "AppControl Sampler"
}
