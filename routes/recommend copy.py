from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from services.image_service import ImageService, ImageUploader
from services.serpapi_service import SerpApiClient
from utils.response_builder import ResponseBuilder
from utils.Siglip_loader import load_siglip
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
import numpy as np

router = APIRouter()

# spaCy 로드
nlp = spacy.load("en_core_web_sm")

# def extract_object_attributes(text: str):
#     doc = nlp(text)
#     attributes = []
#     for token in doc:
#         if token.pos_ in ("NOUN", "PROPN"):
#             adj = [t.text for t in token.lefts if t.pos_ == "ADJ"]
#             phrase = " ".join(adj + [token.text])
#             if phrase.lower() not in {"background", "scene", "image", "floor", "top", "wall"}:
#                 attributes.append(phrase)
#     return list(set(attributes)) if attributes else text.split()
def extract_object_attributes(text: str):
    doc = nlp(text)
    attributes = []

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        # if not any(bad in phrase.lower() for bad in ["background", "scene", "floor", "wall", "image"]):
        attributes.append(phrase)

    return list(set(attributes))

def select_main_object(model, processor, image, candidates):
    """각 candidate와 이미지 간 cosine similarity 계산 후, 가장 높은 객체 반환"""
    if not candidates:
        return None

    similarities = []
    for cand in candidates:
        inputs = processor(text=[cand], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            sim = outputs.logits_per_image.item()
        similarities.append((cand, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"[SigLIP Similarities] {similarities}")

    return similarities[0][0]


@router.post("/recommend")
async def recommend_from_image(
    image: UploadFile = File(...),
    prompt: str = Form(None)
):
    tmp_path = None
    try:
        tmp_path = await ImageService.save_temp_file(image)
        image_url = await ImageUploader.upload_to_catbox(tmp_path)
        pil_image = Image.open(tmp_path).convert("RGB")

        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        blip_inputs = blip_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = blip_model.generate(
                **blip_inputs,
                max_new_tokens=50,
                num_beams=5,
                do_sample=False
            )
        blip_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"[BLIP Caption] {blip_caption}")

        attributes = extract_object_attributes(blip_caption)
        print(f"[Filtered Attributes] {attributes}")

        model, processor = await load_siglip()
        main_object = select_main_object(model, processor, pil_image, attributes)
        print(f"[Main Object by SigLIP] {main_object}")

        combined_prompt = main_object if prompt is None else f"{prompt} {main_object}"

        results = await SerpApiClient.fetch_shopping_results(combined_prompt, limit=5)

        if isinstance(results, list):
            cleaned_results = []
            for r in results:
                cleaned_results.append({
                    "title": r.get("title"),
                    "image": r.get("thumbnail") or r.get("image"),
                    "link": r.get("link") or r.get("source") or r.get("product_link"),
                    "price": r.get("price")
                })
        else:
            cleaned_results = results.get("shopping_results", [])

        response = ResponseBuilder.build_response(combined_prompt, image_url, cleaned_results)
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"추천 이미지 검색 중 오류 발생: {str(e)}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
