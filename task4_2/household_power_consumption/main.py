import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Загружаем данные с правильным разделителем
df = pd.read_csv(
    "household_power_consumption.csv",
    sep=",",  # <-- Данные разделены запятыми
    na_values="?",  # Преобразуем "?" в NaN
    low_memory=False
)

# Вывод списка столбцов для проверки
print("Колонки в файле:", df.columns)

# Проверяем наличие нужных колонок
if "Date" not in df.columns or "Time" not in df.columns:
    raise ValueError("Ошибка: В файле отсутствуют столбцы 'Date' и 'Time'.")

# Объединение даты и времени в один столбец
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    dayfirst=True  # Формат DD/MM/YYYY
)

# Удаляем старые столбцы
df.drop(columns=["Date", "Time"], inplace=True)

# Преобразуем числовые столбцы в float, если они загружены как строки
numeric_cols = df.columns.difference(["datetime"])
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Заполнение пропущенных значений средним значением по каждому столбцу
df.fillna(df.mean(numeric_only=True), inplace=True)

# Разделение на признаки и целевую переменную
X = df.drop(columns=["Global_active_power"])
y = df["Global_active_power"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))  # Только числовые
X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))

print("Данные успешно обработаны и готовы к обучению модели.")

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
new_data = pd.DataFrame([{
    'Global_reactive_power': 0.418,
    'Voltage': 234.84,
    'Global_intensity': 18.4,
    'Sub_metering_1': 0,
    'Sub_metering_2': 1,
    'Sub_metering_3': 17
}, {
    'Global_reactive_power': 0.436,
    'Voltage': 233.63,
    'Global_intensity': 23,
    'Sub_metering_1': 0,
    'Sub_metering_2': 1,
    'Sub_metering_3': 16
}, {
    'Global_reactive_power': 0.498,
    'Voltage': 233.29,
    'Global_intensity': 23,
    'Sub_metering_1': 0,
    'Sub_metering_2': 2,
    'Sub_metering_3': 17
}, {
    'Global_reactive_power': 0.502,
    'Voltage': 233.74,
    'Global_intensity': 23,
    'Sub_metering_1': 0,
    'Sub_metering_2': 1,
    'Sub_metering_3': 17
}, {
    'Global_reactive_power': 0.528,
    'Voltage': 235.68,
    'Global_intensity': 15.8,
    'Sub_metering_1': 0,
    'Sub_metering_2': 1,
    'Sub_metering_3': 17
}])

# Масштабируем новые данные перед прогнозированием
new_data_scaled = scaler.transform(new_data)

# Прогнозы для новых данных с каждой моделью
for name, model in models.items():
    pred = model.predict(new_data_scaled)
    print(f"Прогнозы для {name}:")
    for i, prediction in enumerate(pred, 1):
        print(f"Прогноз {i}: {prediction:.3f} (Global_active_power)")
