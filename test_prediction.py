import pandas as pd

from src.predict import predict_churn


customer_data = pd.DataFrame([
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 4,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.40,
        "TotalCharges": 318.00
    }
])


churn_probabilities, churn_predictions = predict_churn(
    customer_data
)


print("Churn probability:", churn_probabilities[0])
print("Prediction:", churn_predictions[0])