# Diabetes Predictor

This project uses machine learning to predict the likelihood of diabetes in individuals based on medical data. The model is built using the PIMA Diabetes Dataset and employs data preprocessing techniques and a Support Vector Machine (SVM) classifier to achieve accurate predictions.

## Overview

The Diabetes Predictor project aims to assist in the early detection of diabetes by analyzing medical features such as glucose levels, blood pressure, and insulin levels. The project includes the following key components:

1. Data Collection and Analysis
2. Data Standardization
3. Train-Test Split
4. Model Training
5. Model Evaluation
6. Predictive System
7. Data Visualization

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/your-username/DiabetesPredictor.git
    cd DiabetesPredictor
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Predictor

1. Ensure that the PIMA Diabetes Dataset (`diabetes.csv`) is available in the project directory.

2. Execute the predictor script:
    ```sh
    python predictor.py
    ```

## Dataset

The project utilizes the PIMA Diabetes Dataset, which contains various medical parameters such as:

- Number of Pregnancies
- Plasma Glucose Concentration
- Diastolic Blood Pressure
- Triceps Skinfold Thickness
- 2-Hour Serum Insulin
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age

## Machine Learning Model

The Support Vector Machine (SVM) classifier is used to create the predictive model. The development process includes:

1. **Data Loading and Inspection**:
    - Load the dataset and examine its structure.
    - Compute statistical measures and explore outcome distributions.

2. **Standardizing the Data**:
    - Normalize the feature values to improve model performance.

3. **Splitting the Data**:
    - Divide the dataset into training and testing subsets.

4. **Training the SVM Classifier**:
    - Train the SVM model using the training data.

5. **Evaluating Model Performance**:
    - Assess the accuracy of the model on both training and testing data.

6. **Creating a Predictive System**:
    - Implement a system to predict diabetes status based on user input features.

## Example Usage

```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)



