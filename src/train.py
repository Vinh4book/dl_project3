import os, time, torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from .model import TinyCNN
from tqdm import tqdm

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(epochs=2, batch_size=128, lr=1e-3, out_dir="/workspace/outputs"):
    device = get_device()
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)

    tfm = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root="/workspace/data", train=True, download=True, transform=tfm)
    testset  = torchvision.datasets.CIFAR10(root="/workspace/data", train=False, download=True, transform=tfm)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = TinyCNN(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(trainloader, desc=f"Epoch {ep}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            ep_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"[Train] ep={ep} loss={ep_loss/total:.4f} acc={correct/total:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total   += x.size(0)
    acc = correct/total
    print(f"[Eval] acc={acc:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), path)
    print("Saved:", path)

if __name__ == "__main__":
    EPOCHS     = int(os.getenv("EPOCHS", "2"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
    LR         = float(os.getenv("LR", "1e-3"))
    OUT_DIR    = os.getenv("OUT_DIR", "/workspace/outputs")
    t0=time.time()
    train(EPOCHS, BATCH_SIZE, LR, OUT_DIR)
    print("Duration (s):", round(time.time()-t0,2))
