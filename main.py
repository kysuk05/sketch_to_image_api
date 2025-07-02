from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import generate, health, recommend
from utils.model_loader import load_pipe

app = FastAPI(title="ControlNet Sketch to Image API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(generate.router)
app.include_router(health.router)
app.include_router(recommend.router)

# 파이프라인 전역 선언
pipe = None

@app.on_event("startup")
async def startup_event():
    global pipe
    pipe = await load_pipe()
    generate.pipe = pipe  # 생성 라우터에 주입
    health.pipe = pipe
    recommend.pipe = pipe