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


        # -------------------CLIP文本管线--------------------------------
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



        # ---------------------处理采样器-------------------------------
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")



        # ---------------------处理特征图-------------------------------
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # 如果是T2T任务,特征图是加了噪声的参考图
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            # 图像预处理
            input_image_tensor = input_image.resize((WIDTH,HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor,
                                              dtype=torch.float32,
                                              device=device)
            input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))
            # (H, W, C) -> (B, H, W, C)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (B, H, W, C) -> (B, C, H, W)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Encoder的噪声准备
            # 噪声图: (1, 4, latentH, latentW)
            encoder_noise = torch.randn(latents_shape,
                                        generator=generator,
                                        device=device)
            # 通过VAE Encoder得到了潜空间图
            latents = encoder(input_image_tensor,encoder_noise)
            # 图像加噪
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents,sampler.timesteps[0])

            to_idle(encoder)
        # 非图生图任务
        else:
            # 无需VAE Encoder,直接生成潜空间噪声
            latents = torch.randn(latents_shape,
                                  generator=generator,
                                  device=device)
            



        # ---------------------UNet去噪部分-------------------------------
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        ## 时间步采样部分
        for i,timestep in enumerate(timesteps):
            
            # 获取时间步嵌入 (1,320)
            time_embedding = get_time_embedding(timestep).to(device)
           
            # 获取潜空间特征图输入 (B, 4, LH, LW)
            model_input = latents
            # 若采用CFG,需要复制一份batch
            if do_cfg:
                # 正面负面处理两张图 (B, 4, LH, LW) -> (2*B, 4, LH, LW)
                model_input = model_input.repeat(2, 1, 1, 1)
            
            # 进入噪声预测网络预测噪声
            # 若采用CFG,特征图和文本嵌入都为 Batch * 2
            model_output = diffusion(model_input,context,time_embedding)

            # CFG 公式
            if do_cfg: 
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            # 至此得到了该步的噪声model_output

            # 采样器依据噪声进行去噪
            latents = sampler.step(timestep,latents,model_output)
            
        to_idle(diffusion)



        # 得到去噪后特征图
        #------------------------ VAE Encoder---------------------------
        decoder = models["decoder"]
        decoder.to(device)

        # (B, 4, LH, LW) -> (B, 3, H, W)
        images = decoder(latents)
        to_idle(decoder)


        # 恢复真正图像
        images = rescale(images,(-1,1),(0,255),clamp = True)
        # (B, 3, H, W) -> (B, H, W, C)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu",torch.uint8).numpy()
        return images[0]
                 
def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min,new_max)
    return x
    
# 正余弦时间步嵌入实现
def get_time_embedding(timestep):
    # (160)     |  10000^(-i/d)  i = 0,1,....d - 1
    freqs = torch.pow(10000,-torch.arange(start=0,end=160,dtype=torch.float32) / 160)
    # (1,160)   |  x  =  t * [10000^(-i/d)]

    # timestep 是一个数,需要转换为张量后转换为向量,
    x = torch.tensor([timestep],dtype=torch.float32)[:,None] * freqs[None]
    
    # (1,320)   |  [PE(2i) , PE(2i + 1)] = [cos(x) , sin(x)]
    return torch.cat([torch.cos(x),torch.sin(x)],dim = -1)


