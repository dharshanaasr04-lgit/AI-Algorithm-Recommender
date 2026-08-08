# 🤖 AI Algorithm Recommender

### Machine Learning Based Algorithm Selection System

An AI-powered web application that recommends a suitable Machine Learning algorithm based on the characteristics and requirements of a user's dataset.

The system uses a trained Machine Learning classification model to analyze project requirements and predict the most suitable algorithm.

---

## 📌 Project Overview

Choosing the right Machine Learning algorithm can be difficult, especially for beginners.

Different algorithms perform better depending on factors such as:

- Problem type
- Dataset size
- Number of features
- Number of records
- Accuracy requirements
- Speed requirements
- Interpretability requirements

This project provides an easy-to-use solution where users enter their project details and the trained Machine Learning model recommends a suitable algorithm.

---

## 🎯 Objectives

The main objectives of this project are:

1. To recommend suitable Machine Learning algorithms automatically.
2. To help beginners understand algorithm selection.
3. To use Machine Learning for algorithm recommendation.
4. To provide a simple and user-friendly interface.
5. To display the model's confidence for each recommendation.
6. To explain why the selected algorithm is suitable.

---

## ✨ Features

- 🔹 Classification, Regression and Clustering support
- 🔹 Dataset size selection
- 🔹 Number of features selection
- 🔹 Number of records selection
- 🔹 Accuracy priority
- 🔹 Speed priority
- 🔹 Interpretability priority
- 🔹 AI-based algorithm recommendation
- 🔹 Model confidence percentage
- 🔹 Algorithm description
- 🔹 Simple Streamlit web interface
- 🔹 Deployed online using Streamlit

---

## 🧠 Algorithms Included

The system can recommend the following algorithms:

- Random Forest
- Decision Tree
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Linear Regression
- K-Means

---

## 📊 Dataset

The training dataset contains information about different Machine Learning project requirements.

### Dataset Features

| Feature | Description |
|---|---|
| problem_type | Classification, Regression or Clustering |
| dataset_size | Small, Medium or Large |
| num_features | Number of features |
| num_records | Number of records |
| accuracy_priority | Required accuracy level |
| speed_priority | Required processing speed |
| interpretability | Required model interpretability |
| recommended_algorithm | Target algorithm |

The current training dataset contains **240 records**.

---

## 🤖 Machine Learning Model

The project uses a supervised Machine Learning classification approach.

### Processing Steps

```text
User Input
    ↓
Data Validation
    ↓
Categorical Feature Encoding
    ↓
Feature Processing
    ↓
Trained ML Model
    ↓
Algorithm Prediction
    ↓
Confidence Calculation
    ↓
Recommendation