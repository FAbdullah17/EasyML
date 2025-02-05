# setup.py - Installation Script
from setuptools import setup, find_packages

setup(
    name="EasyML-Python",
    version="0.1.0",
    author="Fahad Abdullah",
    author_email="fahadai.co@gmail.com",
    description="A lightweight automated ML pipeline for preprocessing, training, and evaluation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FAbdullah17/EasyML",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "torch",
        "transformers",
        "xgboost",
        "matplotlib",
        "seaborn",
        "lightgbm",
        "catboost"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries"
    ],
    python_requires='>=3.8',
)