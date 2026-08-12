# Handwritten Digit Classification using ANN

## Project Overview

This project focuses on classifying handwritten digits from the **MNIST dataset** using an **Artificial Neural Network (ANN)**.

The objective is to build a neural network that can recognize handwritten digits from **0 to 9** based on their pixel values.

The project is implemented in a **Jupyter Notebook** using Python and TensorFlow/Keras.

## Dataset

The project uses the **MNIST handwritten digit dataset**, which contains:

- **60,000** training images
- **10,000** test images
- Images of size **28 × 28 pixels**
- Grayscale images
- 10 classes representing digits **0–9**

Each image is represented by 784 pixel values after flattening the 28 × 28 image.

## Technologies Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Project Workflow

1. Load the MNIST dataset
2. Explore and visualize sample images
3. Normalize pixel values
4. Build the Artificial Neural Network
5. Compile the model
6. Train the model
7. Evaluate performance on the test dataset
8. Analyze training and validation performance
9. Generate classification metrics
10. Visualize the confusion matrix
11. Analyze correct and incorrect predictions
12. Save the trained model

## Data Preprocessing

The original pixel values range from **0 to 255**.

To normalize the data, pixel values are divided by 255 so that they fall between **0 and 1**.

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

## Model Architecture

The ANN consists of the following layers:

| Layer | Details |
|---|---|
| Input | 28 × 28 grayscale image |
| Flatten | Converts image into 784 features |
| Hidden Layer 1 | 128 neurons, ReLU activation |
| Hidden Layer 2 | 64 neurons, ReLU activation |
| Output Layer | 10 neurons, Softmax activation |

The overall architecture is:

```text
28 × 28 Image
     ↓
  Flatten
     ↓
784 Features
     ↓
Dense Layer — 128 neurons + ReLU
     ↓
Dense Layer — 64 neurons + ReLU
     ↓
Dense Layer — 10 neurons + Softmax
```

## Model Compilation

The model uses:

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Evaluation Metric:** Accuracy

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Model Training

The ANN is trained for **25 epochs** with **20% of the training data used for validation**.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=25,
    validation_split=0.2
)
```

## Evaluation

The trained model is evaluated using the MNIST test dataset.

The project calculates:

- Test accuracy
- Test loss
- Precision
- Recall
- F1-score
- Confusion matrix

The original project reports a test accuracy of **97.37%**. Actual accuracy may vary slightly depending on the training run.

## Visualizations

The notebook includes visualizations for:

- Sample MNIST images
- Training and validation loss
- Training and validation accuracy
- Confusion matrix
- Correctly classified digits
- Incorrectly classified digits

These visualizations help analyze how the ANN learns and where it makes classification errors.

## Error Analysis

Incorrect predictions are identified by comparing the predicted labels with the actual labels.

This allows the model's mistakes to be visually inspected and helps identify handwritten digits that may be difficult for the ANN to distinguish.

## Model Output

The model produces 10 probability values using the Softmax output layer, corresponding to digits **0–9**.

The predicted digit is selected as the class with the highest probability.

## Project Structure

```text
handwritten-digit-classification/
│
├── Handwritten_Digit_Classification.ipynb
├── README.md
└── mnist_model.h5
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-link>
cd handwritten-digit-classification
```

### 2. Install the required libraries

```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Handwritten_Digit_Classification.ipynb
```

and run the cells sequentially.

## Results

The ANN is able to classify handwritten digits with approximately **97% test accuracy**.

The notebook also provides a detailed view of model performance through training curves, classification metrics, a confusion matrix, and visual inspection of misclassified images.

## Future Improvements

Possible improvements include:

- Experimenting with different numbers of hidden layers and neurons
- Trying different optimizers
- Hyperparameter tuning
- Adding dropout for regularization
- Comparing the ANN with a Convolutional Neural Network (CNN)
- Improving performance through further model experimentation

## License

This project is intended for educational and learning purposes.
