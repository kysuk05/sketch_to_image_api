import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class SerpApiClient:
    @staticmethod
    async def fetch_shopping_results(prompt: str, limit: int = 5) -> list:
        results = []
        async with httpx.AsyncClient() as client:
            params = {
                "engine": "google_shopping", 
                "q": prompt,                    
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
            resp = await client.get("https://serpapi.com/search", params=params)
            if resp.status_code != 200:
                raise Exception(f"SerpAPI 요청 실패: {resp.text}")

            data = resp.json()

            shopping_results = data.get("shopping_results", [])
            if not shopping_results:
                shopping_results = data.get("organic_results", [])

            for item in shopping_results[:limit]:
                if not item.get("thumbnail"):
                    continue
                results.append({
                    "image": item.get("thumbnail"),
                    "title": item.get("title"),
                    "link": item.get("product_link") or item.get("link"),
                    "price": item.get("price", None)
                })

        return results
