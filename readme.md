# 🏦 Customer Churn Prediction - ANN Classification

A machine learning project that predicts customer churn using an Artificial Neural Network (ANN), deployed as an interactive Streamlit web application.
The app can be accessed live [here](https://ann-classification-tf-churn.streamlit.app/).

## 📋 Project Overview

This project builds and deploys a deep learning model to predict whether a bank customer will churn (leave the bank) based on their personal and financial information. The model uses customer features like geography, gender, age, and financial metrics to make predictions.

## 🚀 Features

- **Interactive Web Interface**: Easy-to-use Streamlit app for real-time predictions
- **Deep Learning Model**: TensorFlow/Keras ANN for accurate churn prediction
- **Data Preprocessing**: Includes label encoding, one-hot encoding, and feature scaling
- **Model Persistence**: Pre-trained model and encoders saved for quick deployment

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for building the ANN
- **Streamlit**: Web app framework for deployment
- **Scikit-learn**: Data preprocessing (StandardScaler, LabelEncoder, OneHotEncoder)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 📁 Project Structure

```
ANN-Classification/
├── app.py                      # Streamlit web application
├── experiments.ipynb           # Jupyter notebook for model development
├── model.h5                    # Trained ANN model
├── label_encoder_gender.pkl    # Gender label encoder
├── one_hot_encoder_geo.pkl     # Geography one-hot encoder
├── scaler.pkl                  # Feature scaler
└── readme.md                   # Project documentation
```

## 🏃‍♂️ How to Run

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ANN-Classification
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit tensorflow scikit-learn pandas numpy
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the provided local URL (typically `http://localhost:8501`)

## 🎯 How to Use

1. **Personal Information**: Select customer's geography, gender, and age
2. **Financial Information**: Input customer's financial metrics
3. **Get Prediction**: The model will predict the churn probability in real-time
4. **Interpret Results**: View the likelihood of customer churn

## 🧠 Model Details

- **Architecture**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow/Keras
- **Preprocessing**: 
  - Label encoding for gender
  - One-hot encoding for geography
  - Standard scaling for numerical features

## 📈 Learning Objectives

This project demonstrates:
- Building and training neural networks for classification
- Data preprocessing techniques for ML
- Model serialization and deployment
- Creating interactive web applications with Streamlit
- End-to-end ML project workflow

## 🔧 Development

The model development process is documented in [`experiments.ipynb`](experiments.ipynb), which includes:
- Data exploration and preprocessing
- Model architecture design
- Training and evaluation
- Model export for deployment

---

*This is a learning project focused on understanding ANN classification and model deployment techniques.*