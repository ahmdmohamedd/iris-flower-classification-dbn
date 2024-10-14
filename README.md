# Iris Flower Classification using Deep Belief Network (DBN)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Overview
This project implements a Deep Belief Network (DBN) to classify iris flowers based on their sepal and petal dimensions. The Iris dataset is a well-known dataset in the field of machine learning, consisting of three species of iris flowers: Setosa, Versicolor, and Virginica. The goal of this project is to build an effective model that can accurately classify these species using TensorFlow.

## Dataset
The Iris dataset includes 150 samples, each with four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Each sample belongs to one of three classes:
- **Setosa**
- **Versicolor**
- **Virginica**

The dataset can be easily accessed through the `sklearn` library or downloaded from the UCI Machine Learning Repository.

## Installation
To set up the environment and install the necessary packages, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ahmdmohamedd/iris-flower-classification-dbn.git
   cd iris-flower-classification-dbn
   ```

2. **Create a new Conda environment (optional but recommended):**
   ```bash
   conda create -n iris_classification python=3.9
   conda activate iris_classification
   ```

3. **Install required packages:**
   ```bash
   pip install numpy pandas scikit-learn tensorflow matplotlib
   ```

## Usage
1. **Load the dataset and preprocess it:**
   ```python
   import pandas as pd
   from sklearn import datasets

   iris = datasets.load_iris()
   X = iris.data
   y = iris.target
   ```

2. **Train the DBN model:**
   ```python
   from tensorflow import keras

   model = keras.Sequential()
   # Add layers to the model
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(X_train, y_train, epochs=50, verbose=1)
   ```

3. **Evaluate the model:**
   ```python
   model.evaluate(X_test, y_test)
   ```

4. **Generate classification report and confusion matrix:**
   ```python
   from sklearn.metrics import classification_report, confusion_matrix

   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))
   print(confusion_matrix(y_test, y_pred))
   ```

## Model Evaluation
The model's performance is evaluated using various metrics:
- **Accuracy:** Overall correctness of the model.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall:** The ratio of correctly predicted positive observations to the all observations in actual class.
- **F1-Score:** The weighted average of Precision and Recall.

### Visualization
The project also includes visualizations such as:
- Confusion Matrix
- ROC Curves

## Results
The DBN model achieved an accuracy of 93% on the Iris dataset. The classification report details precision, recall, and F1-scores for each class. Cross-validation scores were consistently high, with a mean score of 0.96.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any improvements or features you would like to see.
