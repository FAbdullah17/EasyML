import pytest
import numpy as np
from easyml.training import Trainer, DeepTrainer, PyTorchTrainer

@pytest.fixture
def sample_data():
    """Creates synthetic data for training tests."""
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    return X_train, y_train, X_test, y_test

def test_trainer(sample_data):
    """Ensure Trainer correctly trains and evaluates a model."""
    X_train, y_train, X_test, y_test = sample_data
    trainer = Trainer("random_forest")
    trained_model = trainer.train(X_train, y_train)
    
    assert trained_model is not None  

def test_deep_trainer(sample_data):
    """Ensure DeepTrainer correctly trains a deep learning model."""
    X_train, y_train, X_test, y_test = sample_data
    deep_trainer = DeepTrainer(input_shape=10, model_type="lstm")
    trained_model = deep_trainer.train(X_train, y_train, X_test, y_test)

    assert trained_model is not None

def test_pytorch_trainer(sample_data):
    """Ensure PyTorchTrainer correctly trains a PyTorch model."""
    X_train, y_train, X_test, y_test = sample_data
    pytorch_trainer = PyTorchTrainer(input_dim=10)
    trained_model = pytorch_trainer.train(X_train, y_train, X_test, y_test)

    assert trained_model is not None
