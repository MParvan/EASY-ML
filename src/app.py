import streamlit as st
import numpy as np
from pipeline import load_dataset, preprocess, extract_features
from helper_utils import load_model, run_inference

st.title("Biosignal / Image Mini-Lab")

task = st.selectbox("Task", ["Image classification","ECG biometrics","EEG classification"])
dataset = st.selectbox("Dataset", ["ECG-ID","Heartprint"])

if st.button("Load sample"):
    X, y = load_dataset(dataset)
    st.write("Loaded:", X.shape, "labels:", set(y) if y is not None else None)

st.subheader("Pipeline options")
if task == "Image classification":
    resize = st.checkbox("Resize to 128x128", value=True)
    normalize = st.checkbox("Normalize", value=True)
else:
    bandpass = st.checkbox("Bandpass 0.5-40 Hz", value=True)

model_choice = st.selectbox("Pretrained model", ["tiny_cnn", "random_forest"])

if st.button("Run inference on sample"):
    X, y = load_dataset(dataset)
    Xp = preprocess(X, task, resize=resize if task.startswith("Image") else None, bandpass=bandpass)
    model = load_model(model_choice)
    preds = run_inference(model, Xp)
    st.write("Predictions:", preds[:10])


# from utils import load_yolo, run_yolo_inference

# st.title("Object Detection Demo")

# uploaded_image = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
# if uploaded_image is not None:
#     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

#     if st.button("Run YOLO Detection"):
#         model = load_yolo("yolov8n.pt")  # or "models/yolo/yolov8n.pt" if offline
#         annotated_img, results = run_yolo_inference(model, uploaded_image)
#         st.image(annotated_img, caption="Detected Objects", use_column_width=True)
#         st.write("Detected:", results[0].names)




from pipeline import run_object_detection
import os

st.title("Object Detection with YOLOv5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    input_path = "input.jpg"
    output_path = "output.jpg"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    run_object_detection(input_path, output_path)
    st.image(output_path, caption="Detected Objects", use_column_width=True)

