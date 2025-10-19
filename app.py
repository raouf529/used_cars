from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from columnTransform import AddDropFeatures, to_string_func


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load("model.pkl")

@app.post("/predict")
def predict(request: dict):
    data = request["data"]
    df = pd.DataFrame([data])

    # Ensure numeric columns are the correct dtype
    numeric_cols = ["km_driven", "engine_capacity(CC)", "make_year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    prediction = pipeline.predict(df)
    return {"prediction": prediction.tolist()}

