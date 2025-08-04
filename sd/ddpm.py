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

    # 去噪过程 q(x_(t-1)|xt, x0)
    def step(self, timestep : int, latents : torch.Tensor , model_output : torch.Tensor):
        # latent ：xt
        # model_output : ε(xt)

        # t
        t = timestep
        # t-1 指上一个时间步
        prev_t = self._get_previous_timestep(t)

        # α¯t
        alpha_prod_t = self.alphas_cumprod[t]
        # α¯(t-1)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        # β¯t
        beta_prod_t = 1 - alpha_prod_t
        # β¯(t-1)
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        # αt
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        # βt
        current_beta_t = 1 - current_alpha_t

        ## 计算x0预测值    公式(15)
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        
        ## 计算μ˜(xt,x0)   公式(7)
        # 计算xt,x0前系数
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** (0.5) * beta_prod_t_prev) / beta_prod_t

        # 计算 μ˜(xt,x0)
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        '''
        去噪公式           公式(6)
        q(x_(t-1)|xt, x0) = N(x_(t-1) ; μ˜(xt,x0) ; (β˜t)*I)  
            I 为单位矩阵 
            (β˜t)*I 代表协方差矩阵,对角线元素分别为β˜t
        重参数化技巧：
            x_(t-1) = μ˜ + (β˜)**0.5 * Noise
        '''
        #  (β˜)**0.5 * Noise 部分
        variance = 0
        if t > 0:
            device = model_output.device
            noise  = torch.randn(model_output.shape, generator=self.generator,device=device,dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        # 重参数化公式    
        # x_(t-1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    

    def add_noise(
        self,
        original_samples : torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # original_samples : 类似x0
        '''
        加噪公式           公式(4)
        q(xt | x0) = N(xt; (α¯t)**0.5*x0 ; (1 - α¯t)*I)  
        重参数化技巧：
            xt = (α¯t)**0.5*x0  + (1 - α¯t)**0.5 * Noise
        '''
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device,dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # (α¯t)**0.5
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
       
        # (1 - α¯t)**0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # 重参数化公式
        noise = torch.randn(original_samples.shape,generator=self.generator,device=original_samples.device,dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

        
    


        







