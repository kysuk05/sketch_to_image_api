## Run with Docker

```bash
docker build -t ai-stack .
docker run --gpus all -p 5000:5000 --env-file .env ai-stack python run.py
