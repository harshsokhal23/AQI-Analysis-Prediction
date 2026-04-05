import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("AQI-INDIA.csv")

df.columns = df.columns.str.lower().str.strip()

df = df.replace("NA", np.nan)

df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')

df = df.dropna(subset=['pollutant_avg'])

df_pivot = df.pivot_table(
    index='city',
    columns='pollutant_id',
    values='pollutant_avg',
    aggfunc='mean'
).reset_index()

df_pivot.columns = df_pivot.columns.str.lower()

required_cols = ['pm2.5', 'pm10', 'no2', 'so2', 'co']
available_cols = [col for col in required_cols if col in df_pivot.columns]

if len(available_cols) > 0:
    df_pivot['aqi'] = df_pivot[available_cols].mean(axis=1)

df_pivot = df_pivot.dropna(subset=['aqi'])

df_pivot[available_cols] = df_pivot[available_cols].fillna(
    df_pivot[available_cols].mean()
)

print(df_pivot.describe())

plt.figure()
plt.hist(df_pivot['aqi'], bins=20)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.heatmap(df_pivot[available_cols + ['aqi']].corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

if 'pm2.5' in df_pivot.columns:
    plt.figure()
    plt.scatter(df_pivot['pm2.5'], df_pivot['aqi'])
    plt.xlabel("PM2.5")
    plt.ylabel("AQI")
    plt.title("PM2.5 vs AQI")
    plt.show()

plt.figure()
sns.boxplot(x=df_pivot['aqi'])
plt.title("AQI Outliers")
plt.show()

X = df_pivot[available_cols]
y = df_pivot['aqi']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest")
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression")
print("RMSE:", rmse_lr)
print("R2 Score:", r2_lr)

plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Random Forest Prediction")
plt.show()

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': available_cols,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
