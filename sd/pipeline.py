import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH  = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt = None,
    input_image = None,
    strength = 0.8, # 图生图添加噪声的强度,强度与自由度成正比
    do_cfg = True,
    cfg_scale = 7.5,
    sampler_name = "ddpm",
    n_inference_steps = 50,
    models = {}, # 模型字典,从其中加载模型
    seed = None,
    device = None,
    idle_device = None,
    tokenizer = None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be bewteen 0 and 1")
        
        if idle_device:
            to_idle = lambda x : x.to(idle_device)
        else:
            to_idle = lambda x : x
        
        # 使用生成器随机生成种子
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # -------------------CLIP文本管线-----------------------
        clip = models["clip"]
        clip.to(device)

        # 处理prompt并根据是否使用CFG做不同处理
        if do_cfg:
            # prompt -> Seq_len = 77 的 token序列
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt],padding = "max_length",max_length = 77
            ).input_ids
            # (B,Seq_len)
            cond_tokens = torch.tensor(cond_tokens,dtype=torch.long,device=device)
            # (B,Seq_len) -> (B,Seq_len,d_embed)
            cond_context = clip(cond_tokens)

            # 对于负面提示词(非条件提示词)同理
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt],padding = "max_length",max_length = 77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens,dtype=torch.long,device=device)
            uncond_context = clip(uncond_tokens)

            # (B,Seq_len,d_embed) + (B,Seq_len,d_embed) -> (2*B ,Seq_len,d_embed)
            context = torch.cat( [cond_context,uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt],padding = "max_length",max_length = 77
            ).input_ids
            tokens = torch.tensor(tokens,dtype=torch.long,device=device)    
            context = clip(tokens)
        
        # clip模型待机至对应设备
        to_idle(clip)

        # ---------------------处理采样器----------------------
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        
        # ---------------------处理特征图----------------------
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # 如果是T2T任务,特征图是加了噪声的参考图
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH,HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor,
                                              dtype=torch.float32,
                                              device=device)
            input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))
            


