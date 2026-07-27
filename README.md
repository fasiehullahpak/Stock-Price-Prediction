# Stock-Price-Prediction
# Data Science project by applying the advance Machine learning model like Deep learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense



df = pd.read_csv("tsla_2014_2023.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())



df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

if 'next_day_close' not in df.columns:
    df['next_day_close'] = df['close'].shift(-1)


df['lag1'] = df['close'].shift(1)
df['lag5'] = df['close'].shift(5)

df['rolling_mean'] = df['close'].rolling(5).mean()
df['rolling_std'] = df['close'].rolling(5).std()

df = df.dropna()



target = 'next_day_close'


features = df.select_dtypes(include=np.number).columns.tolist()
features.remove(target)

X = df[features]
y = df[target]

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]



print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)


print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=100)
xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)


print("Training LSTM...")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['close']])

def create_data(data, step=10):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step])
        y.append(data[i+step])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_data(scaled)

split_lstm = int(len(X_lstm) * 0.8)

X_train_lstm = X_lstm[:split_lstm]
X_test_lstm = X_lstm[split_lstm:]

y_train_lstm = y_lstm[:split_lstm]
y_test_lstm = y_lstm[split_lstm:]

model = Sequential()
model.add(Input(shape=(10,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)

pred_lstm = model.predict(X_test_lstm)

pred_lstm = scaler.inverse_transform(pred_lstm)
y_test_lstm = scaler.inverse_transform(y_test_lstm)


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")

print("\nMODEL PERFORMANCE:")

evaluate(y_test, pred_rf, "Random Forest")
evaluate(y_test, pred_xgb, "XGBoost")
evaluate(y_test_lstm, pred_lstm, "LSTM")



plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual")
plt.plot(pred_rf, label="Random Forest")
plt.plot(pred_xgb, label="XGBoost")

plt.title("Stock Price Prediction (Tesla)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.show()
