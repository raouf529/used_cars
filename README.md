# Used Car Price Prediction

A web application that predicts the price of used cars in the Indian market based on various features using machine learning.

## Project Structure

```
├── api.py                     # Flask API for model predictions
├── columnTransform.py         # Data transformation utilities
├── model.pkl                  # Trained machine learning model
├── pre-owned cars.csv        # Dataset
├── used_cars.ipynb           # Model training notebook
└── used_car_web/            # Frontend web application
    ├── index.html            # Main HTML file
    ├── main.js              # JavaScript functionality
    ├── style.css            # Styling
    └── assets/              # Image assets
```

## Features

- Predicts used car prices based on:
  - Brand (14 major Indian car manufacturers)
  - Model
  - Year of manufacture
  - Fuel type (Petrol/Diesel/CNG/Electric)
  - Transmission type
  - Number of previous owners
  - Spare key availability
  - Insurance status
  - Mileage (km/l)
  - Engine capacity (CC)
  - Registration number

## How to Use

1. Start the Flask API:
   ```bash
   python api.py
   ```

2. Open `used_car_web/index.html` in a web browser

3. Fill in the car details in the form

4. Click "Predict Price" to get the estimated price of the car

## Technical Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- Machine Learning: Scikit-learn
- Data Processing: Pandas, NumPy

## Data Validation

- Registration number validation follows Indian vehicle registration format (e.g., MH12AB1234)
- Year of manufacture restricted between 1980-2025
- All input fields are required for accurate prediction

## Development

The model was trained on Indian used car market data, considering various features that affect car prices in the Indian context. The web interface provides an easy-to-use form for users to input car details and get price predictions.