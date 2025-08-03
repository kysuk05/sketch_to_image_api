from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from PIL import Image
import io, base64, torch

from utils.image_utils import encode_image_to_base64
from utils.detector import detector
from config import RESOLUTION, INFERENCE_STEPS, GUIDANCE_SCALE, NUM_SAMPLES, OUTPUT_PATH

router = APIRouter()
pipe = None

@router.post("/upload")
async def generate_from_sketch(
    sketch: UploadFile = File(...),
    prompt: str = Form("photorealistic image"),
    negative_prompt: Optional[str] = Form(None),
):
    if pipe is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    try:
        image_bytes = await sketch.read()
        sketch_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        control_image = detector(sketch_image, resolution=RESOLUTION)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device)

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            num_samples=NUM_SAMPLES,
            generator=generator,
        )

        base64_image = encode_image_to_base64(output.images[0])
        image_data = base64.b64decode(base64_image.split(",")[-1])

        with open(OUTPUT_PATH, "wb") as f:
            f.write(image_data)

        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "image": f"data:image/png;base64,{base64_image}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 생성 중 오류 발생: {str(e)}")
