# Image Classifier with Python 🖼️🤖

A machine learning application that classifies images into 10 different categories using Convolutional Neural Networks (CNN). Built with TensorFlow/Keras and trained on the CIFAR-10 dataset.

## 🎯 What Does It Do?

This image classifier can identify and categorize images into one of 10 classes:
- ✈️ Plane
- 🚗 Car
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

Simply provide an image, and the model will predict which category it belongs to with a confidence score.

## 👤 Who Is It For?

- Machine learning beginners learning image classification
- Students exploring computer vision and deep learning
- Developers building image recognition features
- Anyone interested in AI and neural networks

## 🚀 How to Use

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohammedhrima/Image-classifier.git
cd Image-classifier
```

2. Install required dependencies:
```bash
pip install opencv-python numpy matplotlib tensorflow
```

### Using the Classifier

#### Option 1: Use Pre-trained Model (Quick Start)

The repository includes a pre-trained model (`image_classifier.model`). To classify an image:

1. Place your image in the project directory (or use the provided sample images)
2. Edit `app.py` and change the image filename:
```python
img = cv.imread('your_image.jpg')  # Replace with your image
```

3. Run the classifier:
```bash
python app.py
```

4. The program will:
   - Display the image
   - Print the prediction (e.g., "Prediction is Car")
   - Show confidence scores for all categories

#### Option 2: Train Your Own Model

If you want to train the model from scratch:

1. Open `app.py`
2. Uncomment the training section (lines marked with triple quotes `"""`)
3. Run the script:
```bash
python app.py
```

4. The training process will:
   - Load 20,000 training images from CIFAR-10
   - Train for 10 epochs (takes 10-30 minutes depending on your hardware)
   - Evaluate accuracy on 4,000 test images
   - Save the trained model as `image_classifier.model`

### Testing with Sample Images

The repository includes sample images for testing:
- `car001.jpg`, `car002.jpg`, `car003.jpg` - Car images
- `horse001.jpg`, `horse002.jpg`, `horse003.jpg` - Horse images

Try classifying these to see the model in action!

## 📊 How It Works

### The Model Architecture

The classifier uses a Convolutional Neural Network (CNN) with the following layers:

1. **Input Layer**: Accepts 32x32 RGB images
2. **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
3. **MaxPooling Layer 1**: 2x2 pooling
4. **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
5. **MaxPooling Layer 2**: 2x2 pooling
6. **Conv2D Layer 3**: 64 filters, 3x3 kernel, ReLU activation
7. **Flatten Layer**: Converts 2D features to 1D
8. **Dense Layer 1**: 64 neurons, ReLU activation
9. **Output Layer**: 10 neurons (one per class), Softmax activation

### Training Process

- **Dataset**: CIFAR-10 (60,000 32x32 color images)
- **Training Set**: 20,000 images
- **Test Set**: 4,000 images
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 10
- **Expected Accuracy**: ~70-75% on test data

### Image Processing Pipeline

1. Load image using OpenCV
2. Resize to 32x32 pixels
3. Convert from BGR to RGB color space
4. Normalize pixel values (divide by 255)
5. Feed to neural network
6. Get prediction probabilities for all 10 classes
7. Select class with highest probability

## 🛠️ Technical Stack

- **TensorFlow/Keras**: Deep learning framework for building and training the CNN
- **OpenCV (cv2)**: Image loading and preprocessing
- **NumPy**: Numerical operations and array manipulation
- **Matplotlib**: Visualization of images and training data
- **CIFAR-10 Dataset**: Standard dataset for image classification

## 📁 Project Structure

```
Image-classifier/
├── app.py                    # Main application script
├── image_classifier.model/   # Pre-trained model (directory)
├── car001.jpg               # Sample car image 1
├── car002.jpg               # Sample car image 2
├── car003.jpg               # Sample car image 3
├── horse001.jpg             # Sample horse image 1
├── horse002.jpg             # Sample horse image 2
├── horse003.jpg             # Sample horse image 3
└── README.md                # This file
```

## 🎓 Learning Outcomes

This project demonstrates:
- Building a CNN from scratch
- Training on a standard dataset (CIFAR-10)
- Image preprocessing techniques
- Model evaluation and accuracy metrics
- Saving and loading trained models
- Making predictions on new images

## 🔧 Customization

### Improve Accuracy

To improve model accuracy, try:
- Increasing training data size (use full 50,000 images)
- Adding more epochs (15-20)
- Adding data augmentation (rotation, flipping, zooming)
- Using a deeper network architecture
- Implementing dropout layers to prevent overfitting

### Classify Your Own Images

To classify custom images:
1. Ensure images are clear and well-lit
2. The subject should be centered and prominent
3. Works best with images similar to CIFAR-10 style
4. Resize happens automatically, but quality matters

## 📈 Performance

- **Training Time**: ~10-30 minutes (CPU), ~2-5 minutes (GPU)
- **Inference Time**: <1 second per image
- **Model Size**: ~2-3 MB
- **Accuracy**: ~70-75% on test set

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with different architectures
- Try other datasets
- Improve preprocessing techniques
- Add more classes
- Implement a web interface

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- TensorFlow and Keras teams for the excellent framework
- OpenCV community for image processing tools
