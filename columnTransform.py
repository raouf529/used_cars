from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from datetime import date

class AddDropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=None, addfearures=True, reference_year=None):
        self.features_to_drop = features_to_drop if features_to_drop is not None else []
        self.addfearures = addfearures
        self.reference_year = reference_year if reference_year else date.today().year
        self.brand_median_engine_ = None  # will be fitted

    def fit(self, X, y=None):
        if 'engine_capacity(CC)' in X.columns and 'brand' in X.columns:
            self.brand_median_engine_ = (
                X.groupby('brand')['engine_capacity(CC)'].median()
            )
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_transformed = X.copy()

        # add one feature
        if self.addfearures and "make_year" in X.columns:
            X_transformed["age"] = self.reference_year - X["make_year"]

        # impute column 'engine_capacity(CC)' without apply/lambda
        if 'engine_capacity(CC)' in X.columns and self.brand_median_engine_ is not None:
            # fill missing values by mapping brand → median
            medians = X_transformed['brand'].map(self.brand_median_engine_)
            X_transformed['engine_capacity(CC)'] = X_transformed['engine_capacity(CC)'].fillna(medians)

        # drop not needed features
        if self.features_to_drop:
            X_transformed = X_transformed.drop(
                columns=[f for f in self.features_to_drop if f in X_transformed.columns]
            )

        return X_transformed

def to_string_func(x):
    return x.astype(str)