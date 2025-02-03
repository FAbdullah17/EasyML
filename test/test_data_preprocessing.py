import unittest
import pandas as pd
import numpy as np
from easyml.data_preprocessing import DataPreprocessor

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Initialize sample data for testing."""
        self.df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        self.preprocessor = DataPreprocessor()

    def test_handle_missing_values(self):
        """Check if missing values are correctly handled."""
        processed_df = self.preprocessor.handle_missing_values(self.df)
        self.assertFalse(processed_df.isnull().any().any(), "Missing values not handled properly.")

    def test_encode_categorical_features(self):
        """Check if categorical encoding works."""
        processed_df = self.preprocessor.encode_categorical_features(self.df)
        self.assertTrue('category' not in processed_df.columns, "Categorical encoding failed.")

    def test_scale_features(self):
        """Ensure feature scaling normalizes data properly."""
        processed_df = self.preprocessor.scale_features(self.df[['feature1', 'feature2']].fillna(0))
        self.assertAlmostEqual(processed_df.mean().mean(), 0, delta=0.1, msg="Scaling failed.")

if __name__ == '__main__':
    unittest.main()
