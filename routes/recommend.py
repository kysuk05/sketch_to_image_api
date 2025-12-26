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

router = APIRouter()
device = "cuda" if torch.cuda.is_available() else "cpu"

nlp = spacy.load("en_core_web_sm")
FOCUS_WEIGHT = 3.0  

def load_blip_large():
    print("[BLIP] Loading BLIP Large with CPU offloading...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        device_map={
            "vision_model": "cpu",
            "text_decoder": device,
            "q_former": device if "q_former" in BlipForConditionalGeneration.__dict__ else "cpu"
        },
        torch_dtype=torch.float16,
    )
    print("[BLIP] Loaded BLIP Large with hybrid placement")
    return model, processor

def extract_object_attributes(text: str):
    doc = nlp(text)
    attributes = [chunk.text.strip() for chunk in doc.noun_chunks]
    return list(set(attributes))

def get_blip_focus_score(blip_model, blip_processor, image, candidates):
    focus_scores = {cand: 0.0 for cand in candidates}
    hooks = []
    attn_maps_per_forward = []

    def hook_fn(module, input, output):
        if hasattr(output, "dim") and output.dim() == 4:
            attn_maps_per_forward.append(output.detach().cpu())

    target_keywords = ["crossattention", "encoder_attn"]
    layer_names = []
    for name, module in blip_model.text_decoder.named_modules():
        if any(k in name.lower() for k in target_keywords):
            hooks.append(module.register_forward_hook(hook_fn))
            layer_names.append(name)

    for cand in candidates:
        cand_tokens = blip_processor.tokenizer(cand, return_tensors="pt")
        cand_ids = cand_tokens.input_ids[0]

        inputs = blip_processor(image, text=cand, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cpu")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            enc_out = blip_model.vision_model(pixel_values=pixel_values).last_hidden_state
        enc_out = enc_out.to(device)

        attn_maps_per_forward.clear()
        with torch.no_grad():
            _ = blip_model.text_decoder(
                input_ids=input_ids,
                encoder_hidden_states=enc_out,
                output_attentions=True
            )

        if attn_maps_per_forward:
            attn_tensor = torch.stack(attn_maps_per_forward)
            attn_tensor = attn_tensor.mean(dim=(0, 2))  # [batch, tgt_len, src_len]

            tgt_tokens = input_ids[0].cpu()
            cand_ids = cand_ids.cpu()
            mask = torch.isin(tgt_tokens, cand_ids)

            # print(f"\n[Candidate Token Focus] {cand}")
            # print(f"  tgt_tokens: {blip_processor.tokenizer.convert_ids_to_tokens(tgt_tokens)}")
            # print(f"  cand_tokens: {blip_processor.tokenizer.convert_ids_to_tokens(cand_ids)}")
            # print(f"  mask: {mask}")

            if mask.any():
                token_focus = attn_tensor[0][mask, :].sum().item()
            else:
                token_focus = attn_tensor[0].sum().item()
            focus_scores[cand] = token_focus
            # print(f"  Focus score: {token_focus}")

    values = torch.tensor(list(focus_scores.values()))
    if values.max() > values.min():
        values = (values - values.min()) / (values.max() - values.min())
        for i, cand in enumerate(focus_scores.keys()):
            focus_scores[cand] = values[i].item()
            # print(f"[Normalized Focus] {cand}: {values[i].item()}")

    return focus_scores

def select_main_object_with_focus(siglip_model, siglip_processor, blip_model, blip_processor, image, candidates):
    if not candidates:
        return None

    focus_scores = get_blip_focus_score(blip_model, blip_processor, image, candidates)
    similarities = []

    siglip_scores = []
    for cand in candidates:
        inputs = siglip_processor(text=[cand], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(siglip_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = siglip_model(**inputs)
            siglip_scores.append(outputs.logits_per_image.item())

    siglip_tensor = torch.tensor(siglip_scores)
    if siglip_tensor.max() > siglip_tensor.min():
        siglip_norm = (siglip_tensor - siglip_tensor.min()) / (siglip_tensor.max() - siglip_tensor.min())
    else:
        siglip_norm = torch.zeros_like(siglip_tensor)

    for i, cand in enumerate(candidates):
        combined = siglip_norm[i].item() + FOCUS_WEIGHT * focus_scores[cand]
        similarities.append((cand, siglip_norm[i].item(), focus_scores[cand], combined))

    similarities.sort(key=lambda x: x[3], reverse=True)
    print("[SigLIP + Focus Scores (normalized)]:")
    for cand, s, f, c in similarities:
        print(f"  {cand:<50} SigLIP={s:.3f}, Focus={f:.3f}, Combined={c:.3f}")

    return similarities[0][0]

@router.post("/recommend")
async def recommend_from_image(image: UploadFile = File(...), prompt: str = Form(None)):
    tmp_path = None
    try:
        tmp_path = await ImageService.save_temp_file(image)
        image_url = await ImageUploader.upload_to_catbox(tmp_path)
        pil_image = Image.open(tmp_path).convert("RGB")

        blip_model, blip_processor = load_blip_large()

        blip_inputs = blip_processor(pil_image, return_tensors="pt")
        blip_inputs = {k: v.to("cpu" if k == "pixel_values" else device) for k, v in blip_inputs.items()}

        with torch.no_grad():
            generated_ids = blip_model.generate(
                **blip_inputs,
                max_new_tokens=50,
                num_beams=5,
                do_sample=False
            )

        caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"[BLIP Caption] {caption}")

        candidates = extract_object_attributes(caption)
        print(f"[Candidates] {candidates}")

        siglip_model, siglip_processor = await load_siglip()

        main_object = select_main_object_with_focus(
            siglip_model, siglip_processor, blip_model, blip_processor, pil_image, candidates
        )
        print(f"[Main Object by SigLIP+Focus] {main_object}")

        final_prompt = main_object if prompt is None else f"{prompt} {main_object}"

        results = await SerpApiClient.fetch_shopping_results(final_prompt, limit=5)
        cleaned = [
            {
                "title": r.get("title"),
                "image": r.get("thumbnail") or r.get("image"),
                "link": r.get("link") or r.get("source") or r.get("product_link"),
                "price": r.get("price")
            }
            for r in results
        ]

        response = ResponseBuilder.build_response(final_prompt, image_url, cleaned)
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 이미지 검색 중 오류 발생: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
