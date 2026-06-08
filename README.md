# image-classifier

A small convolutional network that sorts an image into one of the ten CIFAR-10
categories: plane, car, bird, cat, deer, dog, frog, horse, ship, truck. Built with
TensorFlow/Keras.

The trained model isn't committed, so train it once and then classify any image. A
few sample photos (`car001.jpg`, `horse001.jpg`, ...) are included to test against.

## Run

```bash
uv run train.py                 # trains on CIFAR-10, writes image_classifier.keras
uv run classify.py car001.jpg   # prints the predicted category
```
