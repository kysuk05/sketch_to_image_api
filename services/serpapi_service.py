import os
import requests
from dotenv import load_dotenv

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


class SerpApiClient:
    @staticmethod
    def fetch_shopping_results(image_url: str, prompt: str, limit: int = 5) -> list:
        results = []
        start = 0

        while len(results) < limit:
            params = {
                'engine': 'google_shopping',
                'image_url': image_url,
                'q': prompt,
                'api_key': SERPAPI_API_KEY,
                'start': start
            }

            response = requests.get("https://serpapi.com/search", params=params)
            if response.status_code != 200:
                raise Exception(f"SerpAPI 요청 실패: {response.text}")

            data = response.json()
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
