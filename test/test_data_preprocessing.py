import pytest
import pandas as pd
import numpy as np
from easyml.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Creates a sample DataFrame for testing preprocessing."""
    return pd.DataFrame({
        'numerical1': [1.0, 2.5, np.nan, 4.2],
        'numerical2': [10, np.nan, 30, 40],
        'categorical': ['A', 'B', 'A', None]
    })

def test_fit_transform(sample_data):
    """Ensure fit_transform properly handles missing values, encoding, and scaling."""
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.fit_transform(sample_data)

    assert processed_data.shape[0] == sample_data.shape[0]  
    assert not np.isnan(processed_data).any()  