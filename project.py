import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# loading the dataset
data = pd.read_csv("AQI-INDIA.csv")

# making column names clean and consistent
data.columns = data.columns.str.lower().str.strip()

# replacing 'NA' values with proper NaN
data = data.replace("NA", np.nan)

# converting pollutant values into numeric form
data['pollutant_avg'] = pd.to_numeric(data['pollutant_avg'], errors='coerce')

# removing rows where pollutant data is missing
data = data.dropna(subset=['pollutant_avg'])

# reshaping data so each pollutant becomes a column
pivot_data = data.pivot_table(
    index='city',
    columns='pollutant_id',
    values='pollutant_avg',
    aggfunc='mean'
).reset_index()

# again making sure column names are clean
pivot_data.columns = pivot_data.columns.str.lower()

# selecting main pollutants we need
pollutants = ['pm2.5', 'pm10', 'no2', 'so2', 'co']
features = [col for col in pollutants if col in pivot_data.columns]

# calculating AQI as average of available pollutants
if len(features) > 0:
    pivot_data['aqi'] = pivot_data[features].mean(axis=1)

# removing rows where AQI couldn't be calculated
pivot_data = pivot_data.dropna(subset=['aqi'])

# filling remaining missing values using column mean
pivot_data[features] = pivot_data[features].fillna(
    pivot_data[features].mean()
)

# basic summary of dataset
print(pivot_data.describe())

# checking how AQI values are distributed
plt.figure()
plt.hist(pivot_data['aqi'], bins=20)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Count")
plt.show()

# seeing relationship between pollutants
plt.figure()
sns.heatmap(pivot_data[features + ['aqi']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# simple relation between PM2.5 and AQI
if 'pm2.5' in pivot_data.columns:
    plt.figure()
    plt.scatter(pivot_data['pm2.5'], pivot_data['aqi'])
    plt.xlabel("PM2.5")
    plt.ylabel("AQI")
    plt.title("PM2.5 vs AQI")
    plt.show()

# checking if there are outliers in AQI
plt.figure()
sns.boxplot(x=pivot_data['aqi'])
plt.title("AQI Spread")
plt.show()

# separating input features and output
X = pivot_data[features]
y = pivot_data['aqi']

# splitting data into training and testing parts
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# training Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# making predictions using Random Forest
rf_pred = rf_model.predict(x_test)

# training Linear Regression model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# predictions using Linear Regression
lr_pred = lr_model.predict(x_test)

# evaluating Random Forest performance
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# evaluating Linear Regression performance
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\nLinear Regression Results")
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

# comparing actual vs predicted values
plt.figure()
plt.scatter(y_test, rf_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (Random Forest)")
plt.show()

# creating a comparison dataframe
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': rf_pred
}).reset_index(drop=True)

comparison = comparison.sort_values(by='Actual')

# plotting trend comparison
plt.figure()
plt.plot(comparison['Actual'].values, label='Actual AQI')
plt.plot(comparison['Predicted'].values, linestyle='dashed', label='Predicted AQI')
plt.xlabel("Samples")
plt.ylabel("AQI")
plt.title("Actual vs Predicted Trend")
plt.legend()
plt.show()

# checking which pollutant affects AQI the most
importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
