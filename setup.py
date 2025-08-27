from setuptools import setup, find_packages

setup(
    python_requires=">=3.9",

    name="diabetes-analysis-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.37.1",
        "pandas>=2.0.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.3.0",
        "imbalanced-learn>=0.12.0",
        "matplotlib>=3.7.0",
        "joblib>=1.3.0"
    ],
    description="Streamlit multi-page scaffold for data analysis and ML (Diabetes/Penguins/Custom).",
    author="",
)