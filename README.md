# 🚗 Car Price Prediction

A Machine Learning web application that predicts **car prices** based on vehicle features such as brand, model, year, mileage, fuel type, and transmission.

The project uses a **Deep Neural Network (DNN)** and provides an interactive **Streamlit** interface for making predictions.

## 🎯 Objective

Build an easy-to-use web application that allows users to enter car information and get an estimated car price.

## 🔄 Workflow

```text
Dataset
   ↓
Data Cleaning & EDA
   ↓
Feature Engineering
   ↓
Encoding & Scaling
   ↓
DNN Model Training
   ↓
Model Evaluation
   ↓
Streamlit Application
   ↓
Car Price Prediction
```

## 🧠 Model

* Deep Neural Network (DNN)
* Optimizer: Adam
* Loss: MSE
* Metric: MAE
* Early Stopping

## 🌐 Streamlit Application

The Streamlit application provides an interactive interface where users can:

* Enter car information
* Select categorical features
* Input numerical features
* Get a predicted car price
* Use the trained model directly through the web interface

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

## 📁 Project Structure

```text
Car-Price-Prediction/
│
├── dataset/
├── notebooks/
├── models/
├── preprocessing/
├── app.py
├── requirements.txt
└── README.md
```

## ▶️ Run the Application

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Evaluation

The model performance is evaluated using:

* **MAE**
* **MSE**
* **RMSE**
* **R² Score**

## 🚀 Future Improvements

* Improve model accuracy
* Add more car features
* Improve the Streamlit UI
* Deploy the application online
* Add more Machine Learning models for comparison

## 👨‍💻 Project

An academic **Machine Learning project** combining data analysis, preprocessing, Deep Learning, and Streamlit deployment to create an interactive **Car Price Prediction** application.

⭐ If you like the project, consider giving the repository a star!
