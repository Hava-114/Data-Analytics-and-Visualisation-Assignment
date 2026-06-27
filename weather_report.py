import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create Mock Historical Weather Dataset (30 days of data)
# In real scenarios, you would load a CSV file containing columns like Date, Temp, Humidity, etc.
np.random.seed(42)
dates = pd.date_range(start="2026-05-01", periods=30)

data = {
    'Date': dates,
    'MaxTemp': [28, 29, 31, 30, 32, 33, 31, 29, 27, 26, 28, 30, 32, 34, 35, 33, 31, 30, 29, 28, 30, 31, 33, 34, 32, 30, 29, 31, 32, 33],
    'Humidity': np.random.randint(50, 85, size=30),
    'WindSpeed': np.random.randint(5, 25, size=30),
    'Pressure': np.random.randint(1005, 1018, size=30),
    'Rainfall': np.random.choice([0.0, 0.0, 1.2, 5.4, 0.0, 12.1, 0.0, 0.0, 2.1, 0.0], size=30)
}

df = pd.DataFrame(data)

# ========================================================
# 2. FEATURE ENGINEERING: Creating Target Variable & Lags
# ========================================================
# To predict tomorrow's weather, we need to shift the MaxTemp column back by 1 day.
# This "Tomorrow_MaxTemp" becomes our target variable (Y).
df['Tomorrow_MaxTemp'] = df['MaxTemp'].shift(-1)

# Create a lag feature: Yesterday's temperature (helps the model notice trends)
df['Yesterday_MaxTemp'] = df['MaxTemp'].shift(1)

# Drop rows with NaN values created due to shifting
df = df.dropna().reset_index(drop=True)

# Define Features (X) and Target (Y)
features = ['MaxTemp', 'Yesterday_MaxTemp', 'Humidity', 'WindSpeed', 'Pressure', 'Rainfall']
X = df[features]
y = df['Tomorrow_MaxTemp']

# ========================================================
# 3. TRAIN-TEST SPLIT & MODEL TRAINING
# ========================================================
# Split the historical dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Machine Learning model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ========================================================
# 4. MODEL EVALUATION
# ========================================================
predictions = model.predict(X_test)

print("=== Weather Predictor Evaluation ===")
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
print(f"Model Accuracy (R² Score): {r2 * 100:.2f}%")
print("-" * 40 + "\n")

# ========================================================
# 5. PREDICTING THE NEXT DAY'S WEATHER (INFERENCE)
# ========================================================
# We pull the most recent record from our dataset to predict tomorrow's temperature
latest_recorded_day = df.iloc[-1]

# Format it to match our feature structure
latest_features = pd.DataFrame([{
    'MaxTemp': latest_recorded_day['MaxTemp'],
    'Yesterday_MaxTemp': latest_recorded_day['Yesterday_MaxTemp'],
    'Humidity': latest_recorded_day['Humidity'],
    'WindSpeed': latest_recorded_day['WindSpeed'],
    'Pressure': latest_recorded_day['Pressure'],
    'Rainfall': latest_recorded_day['Rainfall']
}])

predicted_temp = model.predict(latest_features)[0]

print("=== TOMORROW'S WEATHER FORECAST ===")
print(f"Latest Recorded Temp (Today): {latest_recorded_day['MaxTemp']}°C")
print(f"Predicted Max Temp for Tomorrow: {predicted_temp:.1f}°C")
print("-" * 40 + "\n")

# ========================================================
# 6. VISUALIZE PREDICTIONS VS ACTUAL
# ========================================================
plt.figure(figsize=(8, 4))
sns.scatterplot(x=y_test, y=predictions, color='blue', edgecolor='black', s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Tomorrow Max Temperatures')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()
