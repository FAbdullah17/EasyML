from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from easyml.models import ModelBuilder, DeepLearningModels, PyTorchModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Trainer:
    def __init__(self, model_name="random_forest", hyperparameter_tuning=False, param_grid=None):
        self.model_name = model_name
        self.hyperparameter_tuning = hyperparameter_tuning
        self.param_grid = param_grid
        self.model = ModelBuilder(model_name=self.model_name).model

    def train(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        if self.hyperparameter_tuning and self.param_grid:
            grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best Params: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Model Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return self.model

class DeepTrainer:
    def __init__(self, input_shape, model_type="lstm", epochs=10, batch_size=32):
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = DeepLearningModels(input_shape, model_type).model

    def train(self, X_train, y_train, X_test, y_test, callbacks=None):
        """Train the deep learning model."""
        if callbacks is None:
            callbacks = []  # Default empty list if no callbacks provided

        self.model.fit(
            X_train, y_train, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_data=(X_test, y_test), 
            verbose=1,
            callbacks=callbacks  # <-- Pass callbacks here
        )
        return self.model

class PyTorchTrainer:
    def __init__(self, input_dim, epochs=10, learning_rate=0.001):
        self.model = PyTorchModel(input_dim)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, X_train, y_train, X_test, y_test):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()
            if epoch % 5 == 0:
                print(f"Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}")
        
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
            predictions = (predictions > 0.5).float()
            accuracy = (predictions == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"Test Accuracy: {accuracy:.4f}")
        return self.model
    