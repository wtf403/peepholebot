FROM jrottenberg/ffmpeg:7.1-ubuntu2404-edge

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m venv /opt/venv && \
    /bin/bash -c "source /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt"

COPY . .
