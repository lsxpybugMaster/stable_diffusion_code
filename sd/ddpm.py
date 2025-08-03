# 采样器部分
import torch
import numpy as np

class DDPMSampler:

    def __init__(
        self,
        generator : torch.Generator,
        num_training_steps = 1000,
        beta_start : float = 0.00085,
        beta_end   : float = 0.0120
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
    ):
        
        # 非线性插值 
        # 噪声调度序列β [βt]
        self.betas = torch.linspace(beta_start ** 0.5,
                                    beta_end ** 0.5,
                                    num_training_steps,
                                    dtype=torch.float32) ** 2
        # 保留系数序列α [αt]  αt = 1 - βt   
        self.alphas = 1.0 - self.betas

        # α累乘   α¯t = α1*α2*α3*....*αt
        self.alphas_cumprod = torch.cumprod(self.alphas,dim = 0)

        self.one = torch.tensor(1.0)

        
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        
        # 时间步序列 降序序列指导去噪
        self.timesteps = torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())


    #  确定推理时的时间步序列(跳步)
    def set_inference_timesteps(self,num_inference_steps = 50):
       
        self.num_inference_steps = num_inference_steps

        # 1000步训练 / 50步推理 = 20步的进行跳跃
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        
        # [999,998,997,...,2,1,0] => [980,960,940,...,40,20,0]
        timesteps = (np.arange(0,num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        # 时间步序列变为推理的时间步
        self.timesteps = torch.from_numpy(timesteps)


    # 根据当前时间步获取上一时间步 [980 -> 960]
    def _get_previous_timestep(self, timestep : int) -> int:

        prev_t  = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    # 计算β˜t
    def _get_variance(self,timestep : int) -> torch.Tensor:
        # β˜t = [(1 - α¯(t-1)) / (1 - α¯t)] * βt
       
        # t-1
        prev_t = self._get_previous_timestep(timestep)
        # α¯t
        alpha_prod_t = self.alphas_cumprod[timestep]
        # α¯(t-1)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        # βt = 1 - αt
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        # β˜t = [(1 - α¯(t-1)) / (1 - α¯t)] * βt
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # 确保值不会为0,因为该数要取log
        variance = torch.clamp(variance,min = 1e-20)

        return variance

    # 设置图生图的加噪强度
    def set_strength(self, strength = 1):
        # 计算需要跳跃的推理时间步
        # strength = 1 时 start_step = 0, 相当于不进行任何跳跃,完整去噪
        # strength = 0 时 start_step = infer_steps, 相当于不进行任何去噪
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)

        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
