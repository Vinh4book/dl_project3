# DL Project 3 (Train → Test → Web) — ENTRYPOINT at /opt/app

- Image name/tag pinned to **docker.io/vinhalex/project3**.
- App code lives at **/opt/app** (not shadowed by Runpod volumes).
- Mount your Runpod network volume to **/workspace** for data/outputs.

## Local quick start
```
docker build -t vinhalex/project3:latest .
docker run --rm -it -e MODE=train -v $PWD/outputs:/workspace/outputs -v $PWD/data:/workspace/data vinhalex/project3:latest
docker run --rm -it -e MODE=web -p 8000:8000 -v $PWD/outputs:/workspace/outputs vinhalex/project3:latest
# open http://localhost:8000/health
```

## Runpod
- Image: `docker.io/vinhalex/project3:latest` (hoặc `:SHA` sau khi CI build)
- Start Command: *(leave empty)*
- ENV: `MODE=web|train|test` (+ `EPOCHS,BATCH_SIZE,LR`)
- Port: 8000
