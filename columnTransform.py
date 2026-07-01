from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from datetime import date

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        self.model_medians_ = {}
        self.brand_fuel_medians_ = {}
        self.brand_medians_ = {}
        self.overall_median_ = None

    def fit(self, X, y=None):
        self.model_medians_ = X.groupby('model')['engine_capacity(CC)'].median()
        self.brand_fuel_medians_ = X.groupby(['brand', 'fuel_type'])['engine_capacity(CC)'].median()
        self.brand_medians_ = X.groupby('brand')['engine_capacity(CC)'].median()
        self.overall_median_ = X['engine_capacity(CC)'].median()
        return self

    def transform(self, X):
        X_transformed = X.copy()

        if 'engine_capacity(CC)' in X_transformed.columns:
            def impute_row(row):
                if not pd.isna(row['engine_capacity(CC)']):
                    return row['engine_capacity(CC)']
                # Step 1: model
                if row['model'] in self.model_medians_ and not pd.isna(self.model_medians_[row['model']]):
                    return self.model_medians_[row['model']]
                # Step 2: brand + fuel
                key = (row['brand'], row['fuel_type'])
                if key in self.brand_fuel_medians_.index and not pd.isna(self.brand_fuel_medians_[key]):
                    return self.brand_fuel_medians_[key]
                # Step 3: brand
                if row['brand'] in self.brand_medians_ and not pd.isna(self.brand_medians_[row['brand']]):
                    return self.brand_medians_[row['brand']]
                # Step 4: overall
                return self.overall_median_

            X_transformed['engine_capacity(CC)'] = X_transformed.apply(impute_row, axis=1)

        if 'transmission' in X_transformed.columns:
            X_transformed['transmission'] = X_transformed['transmission'].apply(lambda x: 1 if x == 'Automatic' else 0)

        if 'spare_key' in X_transformed.columns:
            X_transformed['spare_key'] = X_transformed['spare_key'].apply(lambda x: 1 if x == 'Yes' else 0)

        X_transformed = X_transformed.drop(columns=self.columns_to_drop, errors='ignore')
        return X_transformed

def to_string_func(x):
    return x.astype(str)