from pathlib import Path
import json

import joblib
import torch
from torch import nn


# -----------------------------
# Paths
# -----------------------------

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_DIRECTORY / "model"


# -----------------------------
# Load preprocessing
# -----------------------------

preprocessor = joblib.load(
    MODEL_DIRECTORY / "preprocessor.joblib"
)


# -----------------------------
# Recreate neural network
# -----------------------------

churn_neural_network = nn.Sequential(
    nn.Linear(45, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)


# -----------------------------
# Load trained model weights
# -----------------------------

churn_neural_network.load_state_dict(
    torch.load(
        MODEL_DIRECTORY / "churn_model.pth",
        weights_only=True
    )
)

churn_neural_network.eval()


# -----------------------------
# Load prediction threshold
# -----------------------------

with open(MODEL_DIRECTORY / "config.json", "r") as config_file:
    model_config = json.load(config_file)

CHURN_THRESHOLD = model_config["threshold"]


# -----------------------------
# Prediction function
# -----------------------------

def predict_churn(customer_data):

    processed_customer_data = preprocessor.transform(
        customer_data
    )

    if hasattr(processed_customer_data, "toarray"):
        processed_customer_data = processed_customer_data.toarray()

    customer_tensor = torch.tensor(
        processed_customer_data,
        dtype=torch.float32
    )

    with torch.no_grad():

        model_outputs = churn_neural_network(
            customer_tensor
        )

        churn_probabilities = torch.sigmoid(
            model_outputs
        ).squeeze(1).numpy()

    churn_predictions = (
        churn_probabilities >= CHURN_THRESHOLD
    ).astype(int)

    return churn_probabilities, churn_predictions