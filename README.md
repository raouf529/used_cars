# Used Car Price Prediction

A machine learning web app for estimating used car prices in the Indian market, built with a full scikit-learn pipeline, FastAPI backend, and a browser-based frontend.

## Project Overview

```
├── best_model.pkl
├── brand_models.json
├── columnTransform.py
├── main.py
├── pre-owned cars.csv
├── README.md
├── used_cars.ipynb
└── used_car_web/
    ├── index.html
    ├── main.js
    ├── style.css
    └── assets/
```

- `used_cars.ipynb` — EDA, preprocessing, model training, and hyperparameter tuning.
- `pre-owned cars.csv` — raw dataset (2806 rows, 15 columns) of used cars sold in India.
- `columnTransform.py` — custom `Transformer` class for imputation and binary encoding.
- `best_model.pkl` — serialized best pipeline exported with `joblib`.
- `main.py` — FastAPI backend serving the prediction endpoint.
- `used_car_web/` — frontend UI for collecting input and calling the API.
- `brand_models.json` — brand-to-model mapping used by the frontend dropdowns.

---

## Notebook Walkthrough

### 1. Data Loading and Inspection

The raw dataset has 2806 rows and 15 columns. Initial inspection revealed:

- One fully corrupt row (index 2805): all features were NaN except `price = 1,883,558,000`, a clear data entry error. This row was dropped.
- One duplicate row (index 1617), also dropped.
- Final clean dataset: **2804 rows**.

Missing values before cleaning:

| Column | Missing Count | Missing % |
|---|---|---|
| `reg_year` | 2086 | 74.3% |
| `engine_capacity(CC)` | 118 | 4.2% |
| All others | 1 each | ~0.04% |

### 2. Feature Decisions

**Dropped features:**

- `reg_year` — 74% missing; correlation with `make_year` = 0.99, so fully redundant.
- `overall_cost` — correlation with `price` = 0.97, direct leakage (derived from price).
- `title` — concatenation of brand + model + year, redundant with existing columns.
- `reg_number` — registration plate prefix (region code), too noisy with 157 unique values and no reliable signal.

**Kept and engineered:**

- `transmission` → binary (Automatic=1, Manual=0)
- `spare_key` → binary (Yes=1, No=0)
- `has_insurance` → kept as-is (True/False)
- `fuel_type`, `ownership` → one-hot encoded (4 and 3 categories respectively)
- `brand` (15 categories), `model` (818 categories) → target encoded
- `make_year`, `engine_capacity(CC)`, `km_driven` → scaled with `StandardScaler`

### 3. Missing Value Imputation — `engine_capacity(CC)`

Engine capacity was imputed using a hierarchical cascade to handle the 818-model cardinality problem (many singleton models cannot be imputed by model median alone):

1. **Median by `model`** — used when the model has multiple records with known CC values.
2. **Median by `brand` + `fuel_type`** — fallback for singleton or rare models.
3. **Median by `brand`** — secondary fallback.
4. **Overall dataset median** — final fallback.

This logic is implemented inside a custom `Transformer(BaseEstimator, TransformerMixin)` class that correctly separates `fit` (learns medians from training data only) from `transform` (applies them), preventing any leakage during cross-validation.

### 4. Target Transformation

`price` is right-skewed (skewness ≈ 1.40) with a long tail up to ~2.5M INR. It was log-transformed using `np.log1p` before training. All reported metrics are in log-price space; use `np.expm1` to convert predictions back to INR.

### 5. Encoding Strategy

| Feature | Strategy | Reason |
|---|---|---|
| `brand` (15 cats) | Target encoding | Moderate cardinality, price signal per brand |
| `model` (818 cats) | Target encoding | Too many for one-hot; direct price signal |
| `fuel_type` (4 cats) | One-hot | Low cardinality, nominal |
| `ownership` (3 cats) | One-hot | Low cardinality, nominal |
| `transmission`, `spare_key` | Binary | 2 values each |

`sklearn.preprocessing.TargetEncoder` (sklearn ≥ 1.3) is used, which internally applies cross-fitting to prevent target leakage during training.

### 6. Model Comparison

All models evaluated with **5-fold cross-validation on the training set** (`KFold`, `shuffle=True`, `random_state=42`). Metrics are in log-price space.

| Model | MSE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.0343 | 0.1852 | 0.837 |
| Ridge | 0.0342 | 0.1849 | 0.838 |
| Lasso | 0.2119 | 0.4603 | -0.004 |
| Random Forest | 0.0196 | 0.1399 | **0.907** |
| XGBoost | 0.0204 | 0.1429 | 0.903 |

Notes:
- Lasso with default alpha collapses to near-zero predictions — alpha needs tuning before Lasso is useful here.
- Linear and Ridge are nearly identical, suggesting no strong multicollinearity issue.
- Random Forest and XGBoost both reach ~0.90 R², with Random Forest slightly ahead.

### 7. Hyperparameter Tuning — Random Forest

Grid search over the following space using the same 5-fold CV:

```python
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2']
}
```

Best parameters found:

```
max_depth=30, max_features='log2', min_samples_leaf=1,
min_samples_split=2, n_estimators=200
```

Best CV score (MSE in log space): **0.02108** → RMSE ≈ **0.1452**, R² ≈ **0.900**

Marginal improvement over the default Random Forest, indicating the default configuration was already well-suited to this dataset.

---

## Requirements

```bash
python -m pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib
```

For notebook and plotting support:

```bash
python -m pip install notebook matplotlib seaborn
```

---

## Running the API

```bash
python main.py
```

The backend starts at `http://127.0.0.1:8000`.

---

## Using the Frontend

Open `used_car_web/index.html` in a browser, then:

1. Click `Get Started`.
2. Select the car details.
3. Click `Predict Price`.

The frontend sends a POST request to `http://localhost:8000/predict` and displays the predicted price.

---

## API Contract

**Request:**

```json
{
  "data": {
    "brand": "Maruti",
    "model": "Swift",
    "fuel_type": "Petrol",
    "transmission": "Manual",
    "km_driven": 42000,
    "engine_capacity(CC)": 998,
    "ownership": "1st owner",
    "make_year": 2018,
    "has_insurance": "Yes",
    "spare_key": "Yes",
    "reg_number": "MH12AB1234",
    "reg_year": "unknown",
    "overall_cost": 0,
    "title": "unknown"
  }
}
```

**Response:**

```json
{
  "prediction": [123456.78]
}
```

> Note: `reg_year`, `overall_cost`, and `title` are passed as placeholders — the pipeline drops them before prediction.

> Note: The model is trained on `log1p(price)`. The API may return a log-transformed value depending on whether `np.expm1` is applied inside `main.py`.