import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import uvicorn

app = FastAPI(title="ControlNet Sketch to Image API")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 파이프라인 선언
pipe = None

# 모델 로드 함수
@app.on_event("startup")
async def load_model():
    global pipep
    
    try:
        print("모델 로딩 중...")
        # ControlNet 모델 불러오기
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Stable Diffusion과 ControlNet 파이프라인 설정
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        
        # GPU 사용 설정
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("GPU를 사용하여 모델 로드 완료")
            # GPU 메모리 최적화 (GPU 사용 시에만)
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("xformers 메모리 최적화 활성화")
            except:
                print("xformers 최적화를 사용할 수 없습니다. 기본 모드로 실행합니다.")
        else:
            pipe = pipe.to("cpu")
            print("CPU를 사용하여 모델 로드 완료 (처리 속도가 느릴 수 있습니다)")
    
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        pipe = None

# 이미지를 base64로 인코딩하는 함수
def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.post("/upload")
async def generate_from_sketch(
    sketch: UploadFile = File(...),
    prompt: str = Form("photorealistic image"),
    negative_prompt: Optional[str] = Form("low quality, bad drawing, ugly, deformed"),
    guidance_scale: Optional[float] = Form(7.5),
    num_inference_steps: Optional[int] = Form(30),
    seed: Optional[int] = Form(-1)
):
    if pipe is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    try:
        # 스케치 이미지 로드
        contents = await sketch.read()
        sketch_image = Image.open(io.BytesIO(contents))
        
        # 스케치 이미지 전처리
        sketch_image = sketch_image.convert("RGB")
        
        # 시드 설정
        if seed != -1:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        else:
            generator = None
        
        print(f"이미지 생성 중... 프롬프트: '{prompt}', 단계: {num_inference_steps}")
        
        # 이미지 생성
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        # 생성된 이미지 처리
        generated_image = output.images[0]
        
        base64_image = encode_image_to_base64(generated_image)
        
        print("이미지 생성 완료")
        
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "image": f"data:image/png;base64,{base64_image}"
        })
    
    except Exception as e:
        print(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 생성 중 오류 발생: {str(e)}")

@app.get("/check")
async def health_check():
    if pipe is not None:
        return {"status": "healthy", "model_loaded": True, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    return {"status": "healthy", "model_loaded": False}

if __name__ == "__main__":
    print("서버 시작 중... http://localhost:5000에서 접속 가능합니다.")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
