from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from services.image_service import ImageService, ImageUploader
from services.serpapi_service import SerpApiClient
from utils.response_builder import ResponseBuilder
import os

router = APIRouter()


@router.post("/recommend")
async def recommend_from_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    tmp_path = None

    try:

        tmp_path = await ImageService.save_temp_file(image)

        image_url = await ImageUploader.upload_to_catbox(tmp_path)

        results = SerpApiClient.fetch_shopping_results(image_url, prompt, limit=5)

        response = ResponseBuilder.build_response(prompt, image_url, results)
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 이미지 검색 중 오류 발생: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
