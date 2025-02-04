import pytest
import numpy as np
from easyml.models import ModelBuilder

@pytest.fixture
def sample_data():
    """Generates synthetic data for model testing."""
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    return X_train, y_train, X_test

def test_train_model(sample_data):
    """Ensure models can be trained without errors."""
    X_train, y_train, X_test = sample_data
    model_builder = ModelBuilder("random_forest")
    model = model_builder.model
    model.fit(X_train, y_train)

    assert model is not None  # Model should be trained successfully

def test_predict(sample_data):
    """Ensure models can make predictions after training."""
    X_train, y_train, X_test = sample_data
    model_builder = ModelBuilder("random_forest")
    model = model_builder.model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape[0] == X_test.shape[0]  # Check output matches input size
