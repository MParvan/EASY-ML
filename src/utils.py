import torch
import numpy as np

def load_model(name):
    if name == "tiny_cnn":
        # create a tiny dummy model if no file exists
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128*128*3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )
        return model
    # add actual file loading:
    # model = torch.load("models/my_model.pth", map_location='cpu')
    # return model

def run_inference(model, X):
    # convert numpy to torch if needed
    import torch
    if isinstance(X, list):
        # for signals
        return [0 for _ in X]  # dummy preds
    else:
        t = torch.tensor(X, dtype=torch.float32)
        out = model(t)
        preds = out.argmax(dim=1).numpy()
        return preds

