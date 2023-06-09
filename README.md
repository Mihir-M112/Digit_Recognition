# Handwritten Digit Recognition using CNN and TensorFlow

This project aims to develop a robust handwritten digit recognition system using Convolutional Neural Networks (CNN) and the TensorFlow library. The model is trained on the popular MNIST dataset, which consists of thousands of handwritten digit images.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository to your local machine.
```
git clone https://github.com/your-username/handwritten-digit-recognition.git
```
2. Install the required dependencies.
```
pip install -r requirements.txt
```

## Usage

1. Ensure that you have downloaded the MNIST dataset. You can find it [here](http://yann.lecun.com/exdb/mnist/).

2. Run the `train.py` script to train the CNN model.
```
python train.py
```

3. Once the training is complete, you can run the `MNIST_Digit_Recognition.jpynb` script to classify handwritten digit images.
```
MNIST_Digit_Recognition.jpynb image_path
```
Replace `image_path` with the path to the image you want to classify.

## Model Architecture

The CNN model architecture consists of multiple convolutional layers, followed by pooling layers and fully connected layers. This architecture enables the model to learn hierarchical features from the input images, leading to accurate digit recognition.

## Training

The training process involves feeding the model with a batch of handwritten digit images and their corresponding labels. The model then learns to minimize the loss by adjusting its parameters through backpropagation and gradient descent optimization. The training is performed for a specified number of epochs, and the model's performance is continually evaluated on a validation set.

## Evaluation

The model's performance is evaluated on a separate test set of handwritten digit images that were not used during the training phase. The evaluation metrics include accuracy, precision, recall, and F1-score. These metrics provide insights into how well the model generalizes to unseen data and its ability to correctly classify different digits.

## Results

After training the model, it achieves an impressive accuracy of 98%+ on the test set, showcasing its effectiveness in recognizing handwritten digits. The model can be further fine-tuned or improved by adjusting hyperparameters, experimenting with different architectures, or increasing the training data size.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.

