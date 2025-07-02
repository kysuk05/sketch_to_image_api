from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from PIL import Image
import io, base64

router = APIRouter()

@router.post("/recommend")
async def recommend_from_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        # 업로드된 이미지 읽기
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 🔍 여기서 이미지 검색 기능이 들어갈 자리
        # 예시: search_results = image_search(input_image, prompt)

        # 현재는 더미 데이터 반환 (주소는 지도 링크 URL)
        dummy_results = [
            {
                "image": "data:image/png;base64,base64encodedimagedata1",
                "price": "120,000원",
                "address": "https://map.naver.com/v5/entry/place/123456"
            },
            {
                "image": "data:image/png;base64,base64encodedimagedata2",
                "price": "85,000원",
                "address": "https://map.naver.com/v5/entry/place/234567"
            },
            {
                "image": "data:image/png;base64,base64encodedimagedata3",
                "price": None,
                "address": "https://map.naver.com/v5/entry/place/345678"
            },
            {
                "image": "data:image/png;base64,base64encodedimagedata4",
                "price": "45,000원",
                "address": "https://map.naver.com/v5/entry/place/456789"
            },
            {
                "image": "data:image/png;base64,base64encodedimagedata5",
                "price": None,
                "address": "https://map.naver.com/v5/entry/place/567890"
            }
        ]

        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "results": dummy_results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 이미지 검색 중 오류 발생: {str(e)}")
