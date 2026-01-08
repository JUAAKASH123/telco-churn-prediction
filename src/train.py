import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,precision_score,f1_score,recall_score
import os 
import joblib

from preprocessing import load_data_and_preprocessing

df=load_data_and_preprocessing()

y=df['Churn']
x=df.drop('Churn',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

models={
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        class_weight='balanced'),
    'XGBoost': XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        n_estimators=200
    )
}

results=[]

for name, model in models.items():
    model.fit(x_train,y_train)
    preds=model.predict(x_test)

    acc=accuracy_score(y_test,preds)
    prec=precision_score(y_test,preds)
    rec=recall_score(y_test,preds)
    f1=f1_score(y_test,preds)

    results.append({
        'Model':name,
        'Accuracy': acc,
        'Precision':prec,
        'Recall': rec,
        'F1':f1
    })



print(classification_report(y_test,preds))  
print("Model Comparison")
print(pd.DataFrame(results))

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['F1'].idxmax()]
best_name = best_row['Model']

best_model = None
for name, model in models.items():
    if name == best_name:
        best_model = model
        break

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\nBest model '{best_name}' saved to models/best_model.pkl")