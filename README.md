# Customer Churn Predictor

A machine learning application that predicts whether a telecom customer is likely to churn based on customer and billing information.

## Project Overview

This project uses a neural network built with PyTorch to predict customer churn.

The trained model is saved and used outside the training environment through a separate inference pipeline. A Streamlit web application provides an interactive interface for making predictions.

## Features

- Customer churn prediction
- Saved PyTorch model
- Saved preprocessing pipeline
- Standalone prediction script
- Interactive Streamlit web application

## Project Structure

```text
customer-churn/
├── data/
├── model/
│   ├── churn_model.pth
│   ├── config.json
│   └── preprocessor.joblib
├── notebooks/
├── src/
│   └── predict.py
├── test_prediction.py
├── app.py
└── requirements.txt