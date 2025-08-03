import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class SerpApiClient:
    @staticmethod
    async def fetch_shopping_results(image_url: str, prompt: str, limit: int = 5) -> list:
        results = []
        start = 0
        async with httpx.AsyncClient() as client:
            while len(results) < limit:
                params = {
                    'engine': 'google_shopping',
                    'image_url': image_url,
                    'q': prompt,
                    'api_key': os.getenv("SERPAPI_API_KEY"),
                    'start': start
                }
                resp = await client.get("https://serpapi.com/search", params=params)
                if resp.status_code != 200:
                    raise Exception(f"SerpAPI 요청 실패: {resp.text}")

                data = resp.json()
                shopping_results = data.get("shopping_results", [])
                if not shopping_results:
                    break

                for item in shopping_results:
                    if not item.get("thumbnail"):
                        continue
                    results.append({
                        "image": item.get("thumbnail"),
                        "title": item.get("title"),
                        "link": item.get("product_link"),
                        "price": item.get("price", None)
                    })
                    if len(results) >= limit:
                        break
                start += len(shopping_results)

        return results
