import unittest
import numpy as np
from easyml.training import Trainer
from easyml.models import ModelBuilder, DeepLearningModels, PyTorchModel

class TestTraining(unittest.TestCase):

    def setUp(self):
        """Prepare synthetic data for training."""
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_val = np.random.rand(20, 10)
        self.y_val = np.random.randint(0, 2, 20)
        self.model_factory = ModelFactory()
        self.trainer = Trainer()

    def test_train_and_evaluate(self):
        """Ensure training and evaluation work correctly."""
        model = self.model_factory.get_model("random_forest")
        accuracy = self.trainer.train_and_evaluate(model, self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertGreater(accuracy, 0.5, "Training accuracy too low.")

    def test_hyperparameter_tuning(self):
        """Check if hyperparameter tuning runs successfully."""
        model = self.model_factory.get_model("random_forest")
        best_params = self.trainer.hyperparameter_tuning(model, self.X_train, self.y_train)
        self.assertIsInstance(best_params, dict, "Hyperparameter tuning failed.")

if __name__ == '__main__':
    unittest.main()
