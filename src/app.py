from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import joblib

app=FastAPI(
    title="Telco Customer Churn Prediction",
    description="Predicts if a customer will churn based on features"
)

Model_path="models/best_model.pkl"
model=joblib.load(Model_path)
encoders=joblib.load("models/encoders.pkl")
print("Model and Encoder load Successfully")

class CustomerInput(BaseModel):
    gender:str
    SeniorCitizen:int
    Partner:str
    Dependents:str
    tenure:int
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float

@app.get("/")
def home():
    return {"message": "Telco Churn API is running! Use POST /predict or visit /docs"}

@app.post("/predict")
def predict(customer:CustomerInput):
    inputdf=pd.DataFrame([customer.dict()])

    inputdf['TotalCharges']=pd.to_numeric(inputdf['TotalCharges'],errors='coerce')
    inputdf['TotalCharges'].fillna(inputdf['TotalCharges'].median(), inplace=True)

    binary_cols=['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col=='gender':
            inputdf[col]=inputdf[col].map({'Male': 1, 'Female': 0})
        else:
            inputdf[col] = inputdf[col].map({'Yes': 1, 'No': 0})
    
    for col in encoders.keys():
        encoder=encoders[col]
        inputdf[col]=encoder.transform(inputdf[col])

    pred=int(model.predict(inputdf)[0])
    prob=float(model.predict_proba(inputdf)[0][1])

    return{
        "churn_prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(prob, 3)
    }