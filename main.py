from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import uvicorn
from columnTransform import Transformer, to_string_func


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transformer = Transformer(columns_to_drop=["Unnamed: 0"])
pipeline = joblib.load("best_model.pkl")


@app.post("/predict")
def predict(request: dict):
    data = request.get("data")
    if data is None:
        return {"error": "No data provided"}

    df = pd.DataFrame([data])

    # Ensure numeric columns are the correct dtype
    numeric_cols = ["km_driven", "engine_capacity(CC)", "make_year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prefer using the loaded pipeline's internal preprocessing if present.
    try:
        prediction = pipeline.predict(df)
    except Exception:
        # Fallback: apply local transformer then predict
        try:
            df_transformed = transformer.transform(df)
            prediction = pipeline.predict(df_transformed)
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)