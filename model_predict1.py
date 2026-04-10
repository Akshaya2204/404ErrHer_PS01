# model_predict.py

import joblib
import numpy as np
import shap
import requests

# -------------------------------
# CONFIG
# -------------------------------
OPENAI_API_KEY = "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"

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("best_model.pkl")
FEATURE_ORDER = joblib.load("feature_order.pkl")

explainer = shap.TreeExplainer(model)


# -------------------------------
# Prepare Features
# -------------------------------
def prepare_features(input_data):
    features = []

    for feature in FEATURE_ORDER:
        features.append(input_data.get(feature, 0))

    return np.array(features).reshape(1, -1)


# -------------------------------
# Risk Engine
# -------------------------------
def risk_engine(probability):

    if probability < 0.3:
        return "LOW", "APPROVE TRANSACTION"

    elif probability < 0.7:
        return "MEDIUM", "CUSTOMER CONFIRMATION REQUIRED"

    else:
        return "HIGH", "BLOCK TRANSACTION & ALERT FRAUD ANALYST"


# -------------------------------
# Generate AI Investigation Note
# -------------------------------
# -------------------------------
# Generate AI Investigation Note
# -------------------------------
def generate_ai_note(probability, risk_level, shap_values_dict):

    sorted_features = sorted(
        shap_values_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    feature_meaning = {
        "velocity_last_1h": "unusually high number of transactions in a short time",
        "velocity_last_24h": "increased transaction activity over the past day",
        "distance_from_home_km": "transaction occurring far from the user's usual location",
        "transaction_amount": "unusually high transaction amount",
        "is_international": "international transaction pattern",
        "merchant_risk": "interaction with a high-risk merchant",
    }

    reasons_list = []

    for k, _ in sorted_features:
        if k in feature_meaning:
            reasons_list.append(feature_meaning[k])
        else:
            reasons_list.append(k.replace("_", " "))

    reasons = ", ".join(reasons_list)

    prompt = f"""
You are a fraud detection analyst.

A transaction has:
- Fraud probability: {probability:.2f}
- Risk level: {risk_level}

Key behavioral patterns observed: {reasons}

Explain why this transaction is suspicious in a natural, professional way without using technical feature names. Keep it concise (2 lines).
"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5
            }
        )

        result = response.json()

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"This transaction is {risk_level.lower()} risk due to {reasons}."

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def predict_transaction(input_data):

    features = prepare_features(input_data)

    probability = model.predict_proba(features)[0][1]

    risk_level, action = risk_engine(probability)

    # SHAP
    shap_values = explainer(features)

    shap_dict = {
        FEATURE_ORDER[i]: float(val)
        for i, val in enumerate(shap_values.values[0])
    }
    # Get top 5 important features
    sorted_shap = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    # Format for frontend
    shap_output = [
        {
            "feature": k,
            "impact": float(v),
            "direction": "increase_risk" if v > 0 else "decrease_risk"
        }
        for k, v in sorted_shap
    ]

    # AI NOTE
    feature_meaning = {
    "velocity_last_1h": "unusually high number of transactions in a short time",
    "velocity_last_24h": "increased transaction activity over the past day",
    "distance_from_home_km": "transaction occurring far from the user's usual location",
    "transaction_amount": "unusually high transaction amount",
    "is_international": "international transaction pattern",
    "merchant_risk": "interaction with a high-risk merchant",
}
    ai_note = generate_ai_note(probability, risk_level, shap_dict)

    # Alert
    if risk_level == "HIGH":
        alert = "🚨 FRAUD ALERT GENERATED"
    elif risk_level == "MEDIUM":
        alert = "⚠️ VERIFY CUSTOMER"
    else:
        alert = "✅ SAFE TRANSACTION"

    return {
        "fraud_probability": float(probability),
        "risk_level": risk_level,
        "recommended_action": action,
        "alert": alert,
        "ai_investigation_note": ai_note
    }

print("TOKEN:", OPENAI_API_KEY)