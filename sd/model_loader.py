from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

# 官方sd模型的网络名称与我们自定义的不同,因此需做转换
# 同时建立所有模型

def preload_models_from_standard_weights(ckpt_path,device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path,device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'],strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'],strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'],strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'],strict=True)

    return {
        'clip' : clip,
        'encoder' : encoder,
        'decoder' : decoder,
        'diffusion' : diffusion,
    }