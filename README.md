# Handwritten Digit Classification using Artificial Neural Network

A handwritten digit classification project using an **Artificial Neural Network (ANN)** to classify grayscale images of digits from **0 to 9** using the MNIST dataset.

## Overview

The objective of this project is to build and evaluate a feed-forward neural network capable of recognizing handwritten digits. The model uses fully connected Dense layers with **ReLU** activations, **Dropout** regularization, and **Early Stopping** to improve generalization and reduce overfitting.

## Dataset

The project uses the **MNIST handwritten digit dataset**:

* 60,000 training images
* 10,000 test images
* Image size: 28 × 28 pixels
* Grayscale images
* 10 classes: digits 0–9

## Approach

### 1. Data Preprocessing

* Loaded the MNIST dataset using TensorFlow/Keras.
* Visualized sample handwritten digits.
* Checked the original pixel intensity range of **0–255**.
* Normalized pixel values to the range **0–1** by dividing by 255.

### 2. ANN Architecture

The neural network consists of:

```text
Input Image (28 × 28)
        ↓
Flatten
        ↓
Dense (128 neurons, ReLU)
        ↓
Dropout (30%)
        ↓
Dense (64 neurons, ReLU)
        ↓
Dropout (20%)
        ↓
Dense (10 neurons, Softmax)
```

The model contains **109,386 trainable parameters**.

### 3. Model Training

* Optimizer: **Adam**
* Loss function: **Sparse Categorical Crossentropy**
* Metric: **Accuracy**
* Maximum epochs: **25**
* Validation split: **20%**
* Early Stopping based on validation loss
* Patience: **3 epochs**
* Best model weights are restored after training

Dropout and Early Stopping were introduced to control overfitting observed during the initial training experiments.

## Results

The final ANN achieved:

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy | **97.48%** |
| Test Loss     | **0.0783** |

The classification report shows consistently strong performance across all ten digit classes, with precision, recall, and F1-scores generally around **0.97–0.99**.

## Evaluation

Model performance was evaluated using:

* Test accuracy
* Classification report
* Precision
* Recall
* F1-score
* Confusion matrix
* Training vs. validation accuracy
* Training vs. validation loss
* Visualization of correct predictions
* Visualization of incorrectly classified digits

The project also examines the incorrectly classified test images to understand where the ANN struggles.

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**

## Project Structure

```text
├── digit-ann-final.ipynb
└── README.md
```

## How to Run

1. Clone this repository.
2. Open `digit-ann-final.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Install the required libraries if needed:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

4. Run the notebook cells sequentially.
5. The MNIST dataset will be downloaded automatically through TensorFlow/Keras.

## Key Takeaways

* Built a complete ANN pipeline for handwritten digit classification.
* Applied pixel normalization to improve model training.
* Used Dropout and Early Stopping to reduce overfitting.
* Achieved **97.48% accuracy** on the unseen MNIST test set.
* Used multiple evaluation techniques to analyze model performance and classification errors.

## Future Improvements

The current project uses a fully connected ANN. Performance could potentially be improved further by exploring **Convolutional Neural Networks (CNNs)**, which are particularly well suited for image-based classification tasks.
