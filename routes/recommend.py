from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from utils.image_upload import upload_to_catbox
import tempfile
import os
import requests

router = APIRouter()

SERPAPI_API_KEY = "1a0a6ff203243d30332842bd64f7a3ee9f01dce373cb7e1af1b40e3319f6f7e8"

@router.post("/recommend")
async def recommend_from_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        suffix = os.path.splitext(image.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        image_url = upload_to_catbox(tmp_path)

        results = []
        start = 0 

        while len(results) < 5:
            serpapi_params = {
                'engine': 'google_shopping',
                'image_url': image_url,
                'q': prompt,
                'api_key': SERPAPI_API_KEY,
                'start': start
            }

            serpapi_resp = requests.get("https://serpapi.com/search", params=serpapi_params)
            if serpapi_resp.status_code != 200:
                raise Exception(f"SerpAPI 요청 실패: {serpapi_resp.text}")

            serpapi_data = serpapi_resp.json()
            shopping_results = serpapi_data.get("shopping_results", [])
            if not shopping_results:
                break

            for item in shopping_results:
                if not item.get("thumbnail"):
                    continue
                price = item.get("price") if "price" in item else None

                results.append({
                    "image": item.get("thumbnail"),
                    "title": item.get("title"),
                    "link": item.get("product_link"),
                    "price": price
                })

                if len(results) >= 5:
                    break

            start += len(shopping_results)

        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "image_url": image_url,
            "results": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 이미지 검색 중 오류 발생: {str(e)}")

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
