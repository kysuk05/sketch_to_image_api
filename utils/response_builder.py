class ResponseBuilder:
    @staticmethod
    def build_response(prompt: str, image_url: str, results: list) -> dict:
        return {
            "status": "success",
            "prompt": prompt,
            "image_url": image_url,
            "results": results
        }
