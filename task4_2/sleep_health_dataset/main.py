import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Загружаем данные из CSV файла
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Обрабатываем пропущенные значения
df.dropna(inplace=True)

# Преобразуем категориальные переменные в числовые
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})  # Теперь 'Female' также обрабатывается
df['BMI Category'] = df['BMI Category'].map({'Normal': 1, 'Overweight': 2, 'Obese': 3})
df['Sleep Disorder'] = df['Sleep Disorder'].map({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 2})

# Удаляем строки с неопределёнными категориями (если после map() появились NaN)
df.dropna(inplace=True)

# Выбираем признаки и целевую переменную
features = ["Gender", "Age", "Physical Activity Level", "Stress Level", "BMI Category", "Heart Rate", "Daily Steps"]
target = "Sleep Duration"
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

# Прогнозирование для новых данных (5 прогнозов)
new_data_list = [
    {"Gender": 1, "Age": 30, "Physical Activity Level": 50, "Stress Level": 5, "BMI Category": 2, "Heart Rate": 78, "Daily Steps": 7000},
    {"Gender": 0, "Age": 32, "Physical Activity Level": 60, "Stress Level": 6, "BMI Category": 1, "Heart Rate": 72, "Daily Steps": 8000},
    {"Gender": 1, "Age": 35, "Physical Activity Level": 40, "Stress Level": 7, "BMI Category": 3, "Heart Rate": 80, "Daily Steps": 5000},
    {"Gender": 0, "Age": 25, "Physical Activity Level": 70, "Stress Level": 4, "BMI Category": 2, "Heart Rate": 76, "Daily Steps": 9000},
    {"Gender": 1, "Age": 29, "Physical Activity Level": 55, "Stress Level": 5, "BMI Category": 1, "Heart Rate": 70, "Daily Steps": 8500}
]

new_data_df = pd.DataFrame(new_data_list)

# Убеждаемся, что порядок колонок совпадает
new_data_df = new_data_df[features]

# Масштабируем новые данные
new_data_scaled = scaler.transform(new_data_df)

# Прогнозируем для каждого нового набора данных
for name, model in models.items():
    print(f"\nПрогнозы для модели {name}:")
    predictions = model.predict(new_data_scaled)
    for i, pred in enumerate(predictions):
        print(f"Прогноз для набора {i+1}: {pred:.2f} (Sleep Duration)")
