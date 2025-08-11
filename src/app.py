import streamlit as st
import numpy as np
from pipeline import load_dataset, preprocess, extract_features
from utils import load_model, run_inference

st.title("Biosignal / Image Mini-Lab")

task = st.selectbox("Task", ["Image classification","ECG biometrics","EEG classification"])
dataset = st.selectbox("Dataset", ["sample-image","sample-ecg"])

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

