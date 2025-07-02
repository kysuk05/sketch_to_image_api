from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from PIL import Image
import io, base64, torch
from utils.image_utils import encode_image_to_base64
from utils.detector import detector

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
        contents = await sketch.read()
        sketch_image = Image.open(io.BytesIO(contents)).convert("RGB")
        control_image = detector(sketch_image, resolution=512)

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu") if -1 != -1 else None

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            num_samples=5,
            generator=generator
        )

        base64_image = encode_image_to_base64(output.images[0])

        base64_data = base64_image.split(",")[-1]

        # 디코딩해서 파일로 저장
        with open("output.png", "wb") as f:
            f.write(base64.b64decode(base64_data))

        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "image": f"data:image/png;base64,{base64_image}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 생성 중 오류 발생: {str(e)}")