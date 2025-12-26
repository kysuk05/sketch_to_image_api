import torch
from transformers import CLIPProcessor, CLIPModel

async def load_clip():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=dtype
        ).to(device)

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        return model, processor

    except Exception as e:
        print(f"CLIP 로드 실패: {e}")
        return None, None
