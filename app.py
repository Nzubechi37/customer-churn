import streamlit as st
import pandas as pd

from src.predict import predict_churn


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------

st.title("📊 Customer Churn Predictor")

st.write(
    "Enter customer information below to estimate the likelihood "
    "that the customer will churn."
)

st.divider()


# -----------------------------
# Customer Information
# -----------------------------

st.subheader("Customer Information")

column_one, column_two = st.columns(2)

with column_one:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=4
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


with column_two:

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# -----------------------------
# Billing Information
# -----------------------------

st.subheader("Billing Information")

billing_column_one, billing_column_two = st.columns(2)

with billing_column_one:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with billing_column_two:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=79.40
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=318.00
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------

predict_button = st.button(
    "🔍 Predict Churn",
    type="primary",
    use_container_width=True
)


if predict_button:

    customer_data = pd.DataFrame([
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
    ])

    churn_probabilities, churn_predictions = predict_churn(
        customer_data
    )

    churn_probability = churn_probabilities[0]
    churn_prediction = churn_predictions[0]

    st.divider()

    st.subheader("Prediction Result")

    result_column_one, result_column_two = st.columns(2)

    with result_column_one:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    with result_column_two:

        if churn_prediction == 1:
            st.error("⚠️ Customer is predicted to churn.")
        else:
            st.success("✅ Customer is predicted to stay.")