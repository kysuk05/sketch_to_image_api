import os
from dotenv import load_dotenv

load_dotenv()

RESOLUTION = int(os.getenv("RESOLUTION", 512))
INFERENCE_STEPS = int(os.getenv("INFERENCE_STEPS", 30))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", 7.5))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 5))
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output.png")
