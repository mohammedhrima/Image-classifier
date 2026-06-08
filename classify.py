import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2 as cv
import numpy as np
from tensorflow import keras

MODEL_PATH = "image_classifier.keras"
CLASS_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    if len(sys.argv) != 2:
        print("usage: uv run classify.py <image>")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} not found — run `uv run train.py` first")
        return

    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"could not read image: {sys.argv[1]}")
        return
    img = cv.cvtColor(cv.resize(img, (32, 32)), cv.COLOR_BGR2RGB)

    model = keras.models.load_model(MODEL_PATH)
    prediction = model.predict(np.array([img]) / 255.0, verbose=0)
    print(CLASS_NAMES[int(np.argmax(prediction))])


if __name__ == "__main__":
    main()
