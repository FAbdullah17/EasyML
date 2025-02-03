import unittest
import numpy as np
from easyml.models import ModelBuilder, DeepLearningModels, PyTorchModel

class TestModels(unittest.TestCase):

    def setUp(self):
        """Create synthetic data and initialize models."""
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 10)
        self.model_factory = ModelFactory()

    def test_train_model(self):
        """Ensure all models can train without errors."""
        model_names = ["logistic_regression", "random_forest", "svm", "mlp"]
        for model_name in model_names:
            model = self.model_factory.get_model(model_name)
            model.fit(self.X_train, self.y_train)
            self.assertIsNotNone(model, f"{model_name} training failed.")

    def test_predict(self):
        """Verify model prediction output shape."""
        model = self.model_factory.get_model("random_forest")
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0], "Prediction shape mismatch.")

if __name__ == '__main__':
    unittest.main()
