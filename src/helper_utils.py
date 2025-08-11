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



from ultralytics import YOLO

# def load_yolo(model_path="yolov8n.pt"):
#     """
#     Load YOLO model.
#     model_path: can be local file ('models/yolo/yolov8n.pt') or a pretrained name ('yolov8n.pt')
#     """
#     model = YOLO(model_path)
#     return model

# def run_yolo_inference(model, image_path_or_array):
#     """
#     Run YOLO object detection on an image path or numpy array.
#     Returns: PIL image with detections drawn, and raw results.
#     """
#     results = model.predict(source=image_path_or_array, save=False)
#     # results[0].plot() gives numpy array with annotations
#     annotated = results[0].plot()  # RGB numpy
#     from PIL import Image
#     return Image.fromarray(annotated), results




# src/object_detection.py
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io

# Load YOLOv5 model from ultralytics repo (pretrained on COCO)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image_path, output_path):
    # Inference
    results = model(image_path)

    # Render results (in memory, as PIL images)
    results.render()

    # Convert numpy image array to PIL
    rendered_img = Image.fromarray(results.ims[0])

    # Save output
    rendered_img.save(output_path)
    return output_path