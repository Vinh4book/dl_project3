import os, torch
from .model import TinyCNN

class Predictor:
    def __init__(self, weight_path="/workspace/outputs/model.pt", device=None, load_if_exists=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyCNN(num_classes=10).to(self.device)
        self.loaded = False
        if load_if_exists and os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            self.loaded = True
        self.model.eval()

    def predict_tensor(self, x):
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            return logits.softmax(1)
