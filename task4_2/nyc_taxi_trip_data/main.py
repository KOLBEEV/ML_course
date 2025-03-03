import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Загружаем данные
df = pd.read_csv("yellow_tripdata_2015-01.csv")

# Вывод первых 10 строк датасета
print("Первые 10 строк датасета:")
print(df.head(10))

# Преобразуем время начала и окончания поездки в формат datetime
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

# Вычисляем длительность поездки в минутах
df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

# Выбираем признаки и целевую переменную
features = ["passenger_count", "trip_distance", "trip_duration"]
target = "total_amount"
X = df[features]
y = df[target]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2)
    print(f"{name}: MSE={mse:.4f}, R²={r2:.4f}")

# Прогнозирование для новых данных
new_data = pd.DataFrame([{ "passenger_count": 1, "trip_distance": 2.5, "trip_duration": 10 }])
new_data_scaled = scaler.transform(new_data)

for name, model in models.items():
    pred = model.predict(new_data_scaled)
    print(f"Прогноз для {name}: {pred[0]:.2f} (total_amount)")
