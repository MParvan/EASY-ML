# pipeline.py
import numpy as np
from scipy.signal import butter, filtfilt
from torchvision import transforms

def load_dataset(name):
    if name == "sample-image":
        # load a tiny set from data/images using torchvision or PIL
        # here we return dummy arrays for the example
        X = np.random.rand(8, 3, 128, 128)
        y = [0,1,0,1,0,1,0,1]
        return X, y
    if name == "sample-ecg":
        import numpy as np
        X = [np.load('data/ecg/sample1.npy')]
        y = [0]
        return X, y

def bandpass_filter(sig, low=0.5, high=40, fs=250):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def preprocess(X, task, resize=None, bandpass=False):
    if task.startswith("Image"):
        # simple normalization/resizing example
        return X  # assume already in desired shape
    else:
        # apply bandpass to ECG/EEG
        return [bandpass_filter(x) if bandpass else x for x in X]

