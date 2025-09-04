from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
from src.infer import Predictor

app = FastAPI(title="DL Web Inference")
predictor = None

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

@app.on_event("startup")
def _load():
    global predictor
    weight = "/workspace/outputs/model.pt"
    predictor = Predictor(weight_path=weight, load_if_exists=True)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "model_loaded": getattr(predictor, "loaded", False)
    }

@app.get("/")
def root():
    return {"msg": "Hello from FastAPI (GPU ready)",
            "hint": "POST /predict with an RGB image <= 1MB for demo"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(await file.read()).convert("RGB")
    tfm = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    x = tfm(image).unsqueeze(0)  # (1,3,32,32)
    probs = predictor.predict_tensor(x).cpu().squeeze(0)  # (10,)
    topk = torch.topk(probs, k=3)
    result = [{"class": CIFAR10_CLASSES[i], "prob": float(probs[i])} for i in topk.indices.tolist()]
    return {"top3": result}
