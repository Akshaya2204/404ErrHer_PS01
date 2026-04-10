from fastapi import FastAPI
from pydantic import BaseModel
from model_predict import predict_transaction
import random

app = FastAPI()


# -------------------------------
# Input Schema
# -------------------------------
class Transaction(BaseModel):
    transaction_amount: float
    velocity_last_1h: int
    distance_from_home_km: float


# -------------------------------
# ROOT
# -------------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


# -------------------------------
# PREDICT
# -------------------------------
@app.post("/predict")
def predict(data: Transaction):
    try:
        input_data = {
            "transaction_amount": data.transaction_amount,
            "velocity_last_1h": data.velocity_last_1h,
            "distance_from_home_km": data.distance_from_home_km
        }

        result = predict_transaction(input_data)
        return result

    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# RANDOM TRANSACTION (LIVE MODE)
# -------------------------------
@app.get("/random-transaction")
def random_transaction():
    try:
        input_data = {
            "transaction_amount": round(random.uniform(100, 50000), 2),
            "velocity_last_1h": random.randint(0, 20),
            "distance_from_home_km": round(random.uniform(0, 5000), 2)
        }

        result = predict_transaction(input_data)

        return {
            "input": input_data,
            "analysis": result
        }

    except Exception as e:
        return {"error": str(e)}