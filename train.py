import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras

MODEL_PATH = "image_classifier.keras"


def main():
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0

    model = keras.models.Sequential(
        [
            keras.layers.Input((32, 32, 3)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y))

    loss, acc = model.evaluate(test_x, test_y)
    print(f"test loss {loss:.3f}, accuracy {acc:.3f}")

    model.save(MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
