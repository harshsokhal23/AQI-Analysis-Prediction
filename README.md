# 🌍 AQI Analysis & Prediction Project

## 📌 Project Overview

This project focuses on analyzing air pollution data and predicting the Air Quality Index (AQI) using Machine Learning techniques. It helps understand how different pollutants affect air quality.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📊 Features of the Project

✔ Data Cleaning & Preprocessing
✔ Handling Missing Values
✔ Data Visualization (Histogram, Heatmap, Scatter Plot, Boxplot)
✔ AQI Calculation using Pollutants
✔ Machine Learning Models:

* Linear Regression
* Random Forest Regressor
  ✔ Model Evaluation (RMSE & R² Score)

---

## 📂 Dataset

* Dataset used: **AQI-INDIA.csv**
* Contains pollutant data like:

  * PM2.5
  * PM10
  * NO2
  * SO2
  * CO

---

## 🤖 How Prediction Works

The model is trained using pollutant values as input and AQI as output.

👉 Steps:

1. Data is cleaned and processed
2. AQI is calculated
3. Data is split into training and testing sets
4. Machine learning models are trained
5. Prediction is done using `.predict()` function

---

## 📈 Results

* Random Forest model gives better accuracy compared to Linear Regression
* Model performance evaluated using:

  * RMSE (Root Mean Squared Error)
  * R² Score

---

## 🚀 How to Run the Project

1. Install required libraries:

   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Run the Python file:

   ```
   python aqi_analysis.py
   ```

---

## 🔮 Future Improvements

* Deploy as a Web Application
* Real-time AQI prediction
* Use advanced ML models

---

## 👨‍💻 Author

**Harsh Sokhal**

---

⭐ If you like this project, give it a star!
