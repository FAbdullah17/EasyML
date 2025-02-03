import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    def __init__(self, numerical_strategy="mean", categorical_strategy="most_frequent", scaling_method="standard"):
        """
        Initialize the Data Preprocessor.
        :param numerical_strategy: Strategy for imputing numerical columns ('mean', 'median', 'most_frequent').
        :param categorical_strategy: Strategy for imputing categorical columns ('most_frequent', 'constant').
        :param scaling_method: Scaling method for numerical data ('standard' for StandardScaler, 'minmax' for MinMaxScaler).
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
    
    def fit_transform(self, df):
        """
        Apply preprocessing steps to the dataset.
        :param df: Input DataFrame
        :return: Processed NumPy array
        """
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.numerical_strategy)),
            ("scaler", self.scaler)
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.categorical_strategy, fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])
        
        transformed_data = preprocessor.fit_transform(df)
        
        return transformed_data
