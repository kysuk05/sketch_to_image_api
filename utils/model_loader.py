import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

async def load_pipe():
    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        return pipe
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None