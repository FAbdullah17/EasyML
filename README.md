# EasyMLTool



[![PyPI version](https://badge.fury.io/py/easymltool.svg)](https://pypi.org/project/easymltool/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python->=3.6-blue.svg)](https://www.python.org/)

---

## Overview

**EasyMLTool** is a lightweight, automated machine learning pipeline package designed to simplify the entire ML workflow—from data preprocessing and model training to hyperparameter tuning, evaluation, and deployment. Its modular architecture makes it accessible for beginners while offering enough flexibility and extensibility for advanced users. With EasyMLTool, you can build robust machine learning models quickly and efficiently.

---

## Features

- **Automated Data Preprocessing**  
  - Handles missing values, feature scaling, encoding, and splitting via scikit-learn pipelines and column transformers.
- **Multiple Model Support**  
  - Offers a variety of built-in traditional ML models (e.g., RandomForest, SVM, Logistic Regression) as well as deep learning models (e.g., LSTM, CNN, and PyTorch-based models).
- **Hyperparameter Optimization**  
  - Integrated methods (e.g., GridSearchCV) to automatically tune model parameters.
- **Model Evaluation & Testing**  
  - Includes utilities for model evaluation using metrics such as accuracy, F1-score, and more.
  - Comes with comprehensive unit and integration tests.
- **Easy Deployment**  
  - Provides deployment utilities (via Flask) to quickly serve models as APIs.
- **Lightweight & Modular Design**  
  - Clean, modular codebase that facilitates easy integration into various ML workflows.
- **Extensible**  
  - Easily extend or customize components to fit your specific use case.

---

## Installation

You can install **EasyMLTool** directly from PyPI:

```bash
pip install easymltool
```

Alternatively, if you want to install from source:

```bash
git clone https://github.com/FAbdullah17/EasyMLTool.git
cd EasyMLTool
pip install .
```

---

## Usage

Below are several examples to get you started.

### 1. Importing EasyMLTool

```python
from easymltool.data_preprocessing import DataPreprocessor
from easymltool.models import ModelBuilder, DeepLearningModels
from easymltool.training import Trainer, DeepTrainer, PyTorchTrainer
```

### 2. Data Preprocessing

```python
import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# Preprocess the data
preprocessor = DataPreprocessor(numerical_strategy="mean", categorical_strategy="most_frequent", scaling_method="standard")
X_processed = preprocessor.fit_transform(df.drop(columns=['target']))
```

### 3. Training a Traditional ML Model

```python
trainer = Trainer(model_name="random_forest", hyperparameter_tuning=False)
trained_model = trainer.train(X_processed, df['target'])
predictions = trained_model.predict(X_processed[:5])
print("Predictions:", predictions)
```

### 4. Training a Deep Learning Model

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_processed, df['target'], test_size=0.2, random_state=42)

dl_trainer = DeepTrainer(input_shape=X_train.shape[1], model_type="lstm", epochs=50, batch_size=32)
trained_dl_model = dl_trainer.train(X_train, y_train, X_test, y_test)
```

### 5. Saving and Loading Models

```python
dl_trainer.save_model("model.h5")
loaded_model = dl_trainer.load_model("model.h5")
```

### 6. Deploying a Model

```bash
python easymltool/deployment.py
```

---

## Project Structure

```
EasyMLTool/
│── easymltool/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── deployment.py
│   ├── models.py
│   ├── training.py
│   ├── utils.py
│── examples/
│   ├── example_dl.ipynb
│   ├── example_ml.ipynb
│── test/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_models.py
│   ├── test_training.py
│── .gitignore
│── LICENSE
│── pyproject.toml
│── README.md
│── requirements.txt
│── setup.py
```

---

## Contributing

We welcome contributions! If you'd like to contribute:

1. **Fork the repository.**
2. **Create a new feature branch:**
   ```bash
   git checkout -b feature-your-feature
   ```
3. **Commit your changes:**
   ```bash
   git commit -m "Describe your changes"
   ```
4. **Push to your branch:**
   ```bash
   git push origin feature-your-feature
   ```
5. **Open a Pull Request** on GitHub.

---

## Testing

To run the tests, execute:

```bash
pytest test/
```

---

## Deployment

To deploy a trained model:

```bash
python easymltool/deployment.py
```

Use an API client (e.g., Postman) to send POST requests to the API endpoint and retrieve predictions.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Contact

For questions, suggestions, or support, please contact **Fahad Abdullah**:

- **Email:** [fahadai.co@gmail.com](mailto:fahadai.co@gmail.com)
- **GitHub:** [FAbdullah17](https://github.com/FAbdullah17)
- **LinkedIn:** [Fahad Abdullah](https://www.linkedin.com/in/fahad-abdullah-3bb72a270)

---

## Acknowledgments

Special thanks to the open-source community for providing the foundational libraries (such as scikit-learn, TensorFlow, and PyTorch) that made EasyMLTool possible.

---

**Happy Coding with EasyMLTool! 🚀**
