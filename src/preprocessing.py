import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data_and_preprocessing():

    df=pd.read_csv("data/Telco-Customer-Churn.csv")
    # Basic Understanding about data
    print(df.head())
    print(df.columns)        
    print(df.shape) 
    print(df.info())
    print(df.describe())
    print("Churn value counts:", df['Churn'].value_counts())
    # Check number of missing values in each column
    print(df.isnull().sum())
    #checking for duplicated values
    print("Duplicated rows: ",df.duplicated().sum())

    #convert Total chrages object -> numeric
    print("TotalCharge dtype: ",df['TotalCharges'].dtype)
    print("sample values:")
    print(df['TotalCharges'].head())
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

    print("Final missing values:")
    print(df['TotalCharges'].isnull().sum())
    print("TotalCharges dtype:", df['TotalCharges'].dtype)
    # Fill missing values with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    print("Final missing values after imputation:",(df['TotalCharges'].isnull().sum()))

    # Drop customerID (not useful)
    df.drop(columns=["customerID"], inplace=True)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling','Churn']

    for col in binary_cols:
        if col=='gender':
           df[col] = df[col].map({'Male': 1, 'Female': 0})
        else:
            df[col]=df[col].map({'Yes': 1, 'No': 0})

    print(df.info())

    multi_col=['MultipleLines','InternetService','OnlineSecurity',
               'OnlineBackup','DeviceProtection','TechSupport',
               'StreamingTV','StreamingMovies','Contract','PaymentMethod']
    encoders={}
    for col in multi_col:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])
        encoders[col]=le

    # Save all encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/encoders.pkl')
    print("\nPreprocessing COMPLETE!")
    print(df.info())
    return df


if __name__ == "__main__":
    df = load_data_and_preprocessing()
    print("\nSample processed data:")
    print(df.head())
