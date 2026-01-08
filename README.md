Telco Customer Churn Prediction

This project is a complete end-to-end machine learning solution built to predict whether a telecom customer is likely to churn (leave the company).
The focus of this project is not just model accuracy, but clean data handling, clear reasoning, and production-readiness.

Project Overview

I used the well-known IBM Telco Customer Churn dataset to build a practical churn prediction system.
The goal was to identify customers at risk of leaving based on their demographics, service usage, and billing information.

Key highlights

-Careful data cleaning, including handling hidden missing values

-Proper handling of categorical features 

-Comparison of two strong tree-based models: Random Forest and XGBoost

-Addressed class imbalance (only ~26% of customers churn)

-Deployed a working prediction API using FastAPI

-Clean, modular, and readable code with documentation

-Final model chosen: XGBoost, as it provided the best balance between precision, recall, and F1-score.

Dataset

-7,043 customer records

-21 original features (customerID was dropped)

-Target: Churn (Yes / No → converted to 1 / 0)

Main challenge:
-The TotalCharges column contained 11 blank values that appeared as empty strings rather than NaN, requiring explicit conversion and handling.

Model Results:

Model         	Accuracy	Precision	Recall	F1-Score
Random Forest	   0.787	  0.637	    0.458   	0.533
XGBoost	         0.755	  0.535	   0.567	    0.551

-Although Random Forest achieved slightly higher accuracy, XGBoost performed better on recall and F1-score, which is more important for identifying customers at risk of churn.

-Class imbalance was handled using:

-class_weight='balanced' for Random Forest

-scale_pos_weight for XGBoost

Why No Feature Scaling ?

No feature scaling:
Both Random Forest and XGBoost are tree-based models. They do not rely on distance calculations, so feature scaling does not impact performance and would add unnecessary complexity.

Setup & Running the Project

Clone the repository
git clone https://github.com/yourusername/telco-churn-prediction.git

Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate    # Mac/Linux
cd telco-churn-pokak

Install dependencies
pip install -r requirements.txt

Train the model
python src/train.py

Start the API
uvicorn src.app:app --reload
