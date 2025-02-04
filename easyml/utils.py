# utils.py
import joblib
import torch
import numpy as np
from tensorflow.keras.models import load_model

def save_model(model, filename):
    """Save a trained machine learning or deep learning model."""
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), filename)
    else:
        joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model_file(filename, model_type="ml"):
    """Load a saved model."""
    if model_type == "pytorch":
        model = torch.load(filename)
        return model
    elif model_type == "deep":
        return load_model(filename)
    return joblib.load(filename)

def preprocess_input(data):
    """Preprocess input data for ML/DL models."""
    if isinstance(data, list):
        data = np.array(data)
    return data.reshape(1, -1) if len(data.shape) == 1 else data

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return accuracy."""
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    return accuracy

def load_and_predict(model_path, input_data, model_type="ml"):
    """Load a model and make a prediction."""
    model = load_model_file(model_path, model_type)
    input_data = preprocess_input(input_data)
    return model.predict(input_data)