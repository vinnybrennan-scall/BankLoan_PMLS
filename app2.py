from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Bank Loan Defaulter API")

# Task 1-4: Load train, target=DEFAULTER, drop SN, 500 trees
train = pd.read_csv("BANK_LOAN.csv")
FEATURES = ['AGE', 'EMPLOY', 'ADDRESS', 'DEBTINC', 'CREDDEBT', 'OTHDEBT']
X_train = train[FEATURES]
y_train = train['DEFAULTER']

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "bank_model.pkl")

class Customer(BaseModel):
    AGE: float
    EMPLOY: float
    ADDRESS: float
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float

@app.get("/")
def home():
    return {"message": "API running", "train_accuracy": model.score(X_train, y_train)}

# Task 7: /predict endpoint
@app.post("/predict")
def predict(c: Customer):
    data = pd.DataFrame([[c.AGE, c.EMPLOY, c.ADDRESS, c.DEBTINC, c.CREDDEBT, c.OTHDEBT]], columns=FEATURES)
    prob = model.predict_proba(data)[0][1] # Task 5 & 6: probability
    pred = int(prob > 0.5)
    return {
        "default_probability": round(float(prob), 4),
        "default_percentage": f"{prob*100:.2f}%",
        "predicted_class": pred
    }