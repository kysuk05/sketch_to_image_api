import torch
from transformers import AutoProcessor, AutoModel

async def load_siglip():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-384",
            torch_dtype=dtype
        ).to(device)

        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")

        return model, processor

    except Exception as e:
        print(f"SigLIP 로드 실패: {e}")
        return None, None
