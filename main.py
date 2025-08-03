from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import generate, health, recommend
from utils.model_loader import load_pipe

def create_app() -> FastAPI:
    app = FastAPI(title="ControlNet Sketch to Image API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(generate.router)
    app.include_router(health.router)
    app.include_router(recommend.router)

    return app

app = create_app()
pipe = None

@app.on_event("startup")
async def on_startup():
    global pipe
    pipe = await load_pipe()
    generate.pipe = pipe
    health.pipe = pipe
    recommend.pipe = pipe
