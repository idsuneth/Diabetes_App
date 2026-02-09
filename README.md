🩺 Diabetes Prediction System – CIS6005

This repository contains the implementation of a diabetes risk prediction system developed for the CIS6005 – Computational Intelligence module.
The project applies machine learning and deep learning concepts to a real-world healthcare dataset obtained from a Kaggle competition, 
and demonstrates how a trained model can be deployed in a practical application.

📌 Project Overview

Problem: Predict whether a patient is at high risk of diabetes

Dataset: Kaggle – Diabetes Prediction Challenge (Playground Series S5E12)

Task Type: Binary classification

Primary Model: Random Forest Classifier

Secondary Model: Artificial Neural Network (ANN)

Application: Web-based interface built using Streamlit

🧠 Models Used

Random Forest Classifier

Chosen for robustness, interpretability, and strong performance on structured healthcare data

Artificial Neural Network (ANN)

Implemented to evaluate deep learning performance and compare with traditional ML

Model evaluation was performed using accuracy, precision, recall, and F1-score.
The Random Forest model achieved competitive performance with improved interpretability


🖥️ Application Demo

A simple Streamlit web application allows users to input patient details such as:

Age

BMI

Blood pressure

The app then predicts diabetes risk in real time using the trained model.

⚠️ This application is for academic demonstration purposes only and is not intended for clinical use.

📁 Repository Structure
├── app.py                 # Streamlit web application
├── rf_model.pkl           # Trained Random Forest model
├── feature_columns.pkl    # Feature structure used during training
├── README.md              # Project documentation
