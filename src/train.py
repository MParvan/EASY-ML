# train.py
import argparse
from pipeline import load_dataset, preprocess
from helper_utils import save_model, create_model
import torch
from torch.utils.data import DataLoader, TensorDataset

def train_loop(model, loader, epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(epochs):
        for xb, yb in loader:
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {e} loss {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sample-image")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    X, y = load_dataset(args.dataset)
    # convert to tensors, small preprocessing
    import torch
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=2, shuffle=True)
    model = create_model("tiny_cnn")
    train_loop(model, loader, epochs=args.epochs)
    torch.save(model.state_dict(), "models/tiny_cnn.pth")
    print("Saved models/tiny_cnn.pth")

