# EasyML


## Overview

EasyML is a lightweight Python package designed to simplify the machine learning pipeline. It provides tools for automated data preprocessing, model training, hyperparameter tuning, and evaluation, making it easier for users to build and deploy robust ML models efficiently.

## Features

- **Automated Data Preprocessing**: Handles missing values, feature scaling, encoding, and splitting.
- **Multiple Model Support**: Supports deep learning models (LSTM, CNN) and traditional ML models.
- **Hyperparameter Optimization**: Built-in methods for tuning model hyperparameters.
- **Model Deployment**: Deploy trained models with minimal effort.
- **Testing and Validation**: Includes unit tests for all major components.
- **Lightweight and Modular**: Designed for flexibility and easy integration into ML workflows.

## Installation

To install EasyML, use:

```bash
pip install easyml
```

Alternatively, install from source:

```bash
git clone https://github.com/FAbdullah17/EasyML.git
cd EasyML
pip install .
```

## Usage

### 1. Importing EasyML

```python
from easyml.training import DeepTrainer
from easyml.data_preprocessing import preprocess_data
```

### 2. Data Preprocessing

```python
X_train, X_test, y_train, y_test = preprocess_data("dataset.csv")
```

### 3. Training a Model

```python
dl_trainer = DeepTrainer(input_shape=X_train.shape[1], model_type="lstm", epochs=50)
trained_model = dl_trainer.train(X_train, y_train, X_test, y_test)
```

### 4. Saving and Loading a Model

```python
dl_trainer.save_model("model.h5")
loaded_model = dl_trainer.load_model("model.h5")
```

### 5. Model Evaluation

```python
metrics = dl_trainer.evaluate(X_test, y_test)
print(metrics)
```

## Project Structure

```
EasyML/
│── easyml/                            # Core library
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── deployment.py
│   ├── models.py
│   ├── training.py
│   ├── utils.py
│── examples/                          # Example notebooks
│   ├── example_dl.ipynb
│   ├── example_ml.ipynb
│── test/                              # Unit tests
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_models.py
│   ├── test_training.py
│
│── .gitignore
│── LICENSE
│── pyproject.toml
│── README.md
│── requirements.txt
│── setup.py
```

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-xyz`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-xyz`).
5. Open a Pull Request.

## License

EasyML is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## Contact

For questions or suggestions, contact **Fahad Abdullah**:
- Email: [fahadai.co@gmail.com](mailto:fahadai.co@gmail.com)
- GitHub: [FAbdullah17](https://github.com/FAbdullah17)
- LinkedIn: [Fahad Abdullah](https://www.linkedin.com/in/fahad-abdullah-3bb72a270)

---

### 📌 Happy Coding with EasyML! 🚀

