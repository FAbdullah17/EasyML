from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import torch
import torch.nn as nn

class ModelBuilder:
    def __init__(self, model_name="random_forest"):
        """
        Initialize ModelBuilder with the specified model.
        :param model_name: The name of the model to use (string)
        """
        self.model_name = model_name
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate machine learning model."""
        models = {
            "logistic_regression": LogisticRegression(),
            "ridge_classifier": RidgeClassifier(),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "adaboost": AdaBoostClassifier(),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "naive_bayes": GaussianNB(),
            "xgboost": XGBClassifier(),
            "lightgbm": LGBMClassifier(),
            "catboost": CatBoostClassifier(verbose=0),
            "mlp": MLPClassifier(max_iter=500)
        }
        return models.get(self.model_name, RandomForestClassifier())
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        return self.model.score(X_test, y_test)

class DeepLearningModels:
    def __init__(self, input_shape, model_type="lstm"):
        """
        Initialize Deep Learning model.
        :param input_shape: Shape of input data.
        :param model_type: Type of deep learning model ('lstm' or 'nn').
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = self._initialize_dl_model()

    def _initialize_dl_model(self):
        if self.model_type == "lstm":
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(self.input_shape, 1)),
                Dense(1, activation='sigmoid')
            ])
        else:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(self.input_shape,)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the deep learning model."""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)

# PyTorch Deep Learning Model
class PyTorchModel(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x