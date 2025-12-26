FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libturbojpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# pip 최신화
RUN pip install --upgrade pip

# torch 먼저 (index-url 명시)
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# binary deps
COPY requirements.binary.txt .
RUN pip install --no-cache-dir -r requirements.binary.txt

# python deps
# light 먼저
COPY requirements.light.txt .
RUN pip install --no-cache-dir -r requirements.light.txt

# heavy 나중
COPY requirements.heavy.txt .
RUN pip install --no-cache-dir -r requirements.heavy.txt
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl


COPY . .

CMD ["python", "run.py"]
