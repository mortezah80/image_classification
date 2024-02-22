# Image Classification with Convolutional Neural Networks (CNNs)

This repository contains code for building and training convolutional neural networks (CNNs) for image classification tasks using TensorFlow and Keras. The code includes implementations of various CNN architectures such as custom CNN, ResNet50, MobileNetV2, and VGG16.

## Dataset
The code is designed to work with a custom image dataset consisting of five classes: Ak, Ala_Idris, Buzgulu, Dimnit, and Nazli. The dataset is stored in Google Drive and is loaded using Google Colab. Each class contains 100 images.

## Preprocessing
- The images are resized to a standard size of 70x70 pixels.
- Data augmentation techniques such as horizontal flipping, rotation, zooming, brightness adjustment, and shifting are applied to increase the diversity of the training data.
- The dataset is split into training, validation, and test sets.

## Model Architecture
### Custom CNN
- A custom CNN architecture is implemented using TensorFlow and Keras.
- The model consists of convolutional layers with ReLU activation, max-pooling layers, and fully connected layers.
- The final layer uses softmax activation for multi-class classification.

### Autoencoder
- An autoencoder is implemented for denoising images by removing noise from the input data.
- The denoising autoencoder is trained separately and integrated into the main CNN model.

### Transfer Learning
- Transfer learning is applied using pre-trained models such as ResNet50, MobileNetV2, and VGG16.
- These models are fine-tuned for the specific image classification task by adding custom fully connected layers.

## Training
- The models are trained using the Adam optimizer and categorical cross-entropy loss.
- Training progress is monitored using accuracy and loss metrics.
- The best performing model is selected based on validation accuracy.

## Evaluation
- The trained models are evaluated on a separate test set to assess their performance.
- Metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate model performance.
- Confusion matrices are generated to visualize classification results.

## K-Fold Cross Validation
- K-Fold Cross Validation is implemented for robust evaluation of the models.
- The dataset is split into K subsets, and the model is trained and evaluated K times, each time using a different subset for validation.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy
- scikit-learn

## Usage
1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Jupyter notebook `main.ipynb` in a compatible environment (e.g., Google Colab).
