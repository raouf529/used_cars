from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from datetime import date

df = pd.read_csv('pre-owned cars.csv')
df['engine_capacity(CC)'] = pd.to_numeric(df['engine_capacity(CC)'], errors='coerce')
brand_median_engine = df.groupby('brand')['engine_capacity(CC)'].median()

class AddDropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop, addfearures=True, reference_year=None):
        self.features_to_drop = features_to_drop if features_to_drop is not None else []
        self.addfearures = addfearures
        self.reference_year = reference_year if reference_year else date.today().year

    def fit(self, X, y=None):
        return self

    def _impute_engine_capacity(self, row, brand_median_engine):
        if pd.isna(row['engine_capacity(CC)']): 
            return brand_median_engine[row['brand']]
        else:
            return row['engine_capacity(CC)']

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_transformed = X.copy()

        # add one feature
        if self.addfearures and "make_year" in X.columns:
            X_transformed["age"] = self.reference_year - X["make_year"]

        # impute column 'engine_capacity(CC)'
        if 'engine_capacity(CC)' in X.columns:
            brand_median_engine = X_transformed.groupby('brand')['engine_capacity(CC)'].median()
            X_transformed['engine_capacity(CC)'] = X_transformed.apply(
                lambda row: self._impute_engine_capacity(row, brand_median_engine), axis=1
            )

        # drop not needed features
        for feature in self.features_to_drop:
            if feature in X_transformed.columns:
                X_transformed.drop(feature, axis=1, inplace=True)

        return X_transformed
