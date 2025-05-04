import io
import base64
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import uvicorn
from controlnet_aux import LineartDetector

detector = LineartDetector.from_pretrained("lllyasviel/Annotators")

app = FastAPI(title="ControlNet Sketch to Image API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 필요 (프론트)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# 연결된 WebSocket 클라이언트 관리
active_websockets = []

# 전역 파이프라인
pipe = None

# 모델 로드
@app.on_event("startup")
async def load_model():
    global pipe
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# base64 인코딩
def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 진행률 전송 함수
async def send_progress_to_clients(progress: str):
    for ws in active_websockets:
        try:
            await ws.send_text(progress)
        except:
            pass

# Diffusers callback 함수 생성
def create_progress_callback(total_steps):
    def callback(step: int, timestep: int, latents):
        progress = int((step / total_steps) * 100)
        asyncio.create_task(send_progress_to_clients(str(progress)))
    return callback

# WebSocket 엔드포인트
@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

# 이미지 업로드 및 생성
@app.post("/upload")
async def generate_from_sketch(
    sketch: UploadFile = File(...),
    prompt: str = Form("photorealistic image"),
    guidance_scale: Optional[float] = Form(7.5),
    num_inference_steps: Optional[int] = Form(30),
    seed: Optional[int] = Form(-1)
):
    if pipe is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    contents = await sketch.read()
    sketch_image = Image.open(io.BytesIO(contents)).convert("RGB")
    control_image = detector(sketch_image, resolution=512)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed) if seed != -1 else None

    output = pipe(
        prompt=prompt,
        image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_samples=1,
        generator=generator,
        callback=create_progress_callback(num_inference_steps),
        callback_steps=1,
    )

    generated_image = output.images[0]
    base64_image = encode_image_to_base64(generated_image)

    await send_progress_to_clients("done")

    return JSONResponse(content={
        "status": "success",
        "image": f"data:image/png;base64,{base64_image}"
    })

# 프론트 페이지
@app.get("/")
async def main_page():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
