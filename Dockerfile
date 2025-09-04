# Base: PyTorch + CUDA + cuDNN already included
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# ---- App lives in /opt/app (NOT overlaid by Runpod's /workspace volume) ----
WORKDIR /opt/app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg curl ca-certificates &&     rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip &&     pip install --no-cache-dir -r /opt/app/requirements.txt

# Copy source code to /opt/app
COPY src /opt/app/src
COPY app /opt/app/app
COPY tests /opt/app/tests
COPY start.sh /opt/app/start.sh

# Normalize EOL + make executable
RUN sed -i 's/\r$//' /opt/app/start.sh && chmod +x /opt/app/start.sh

# ---- Persist area for Runpod (usually /workspace volume) ----
RUN mkdir -p /workspace/data /workspace/outputs

# Healthcheck for web mode
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3   CMD curl -fsS http://127.0.0.1:8000/health || exit 1

EXPOSE 8000
ENV MODE=web

# Entry: use the script from /opt/app so it won't be hidden by volume mounts
ENTRYPOINT ["/bin/bash", "/opt/app/start.sh"]
