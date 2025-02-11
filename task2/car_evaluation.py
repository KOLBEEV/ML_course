import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import os
import joblib
from datetime import datetime

# Ограничение количества ядер
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Загрузка датасета Car Evaluation
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
data = pd.read_csv(dataset_url, names=columns)

# Преобразование категориальных признаков в числовые
label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Визуализация распределения классов
plt.figure(figsize=(8, 5))
data["class"].value_counts().plot(kind='bar', color='skyblue')
plt.title("Распределение классов автомобилей")
plt.xlabel("Классы")
plt.ylabel("Количество")
plt.xticks(rotation=0)
plt.show()

# Разделение на признаки и целевую переменную
X = data.drop("class", axis=1)
y = data["class"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков для SVM и Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Определение моделей
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Обучение и оценка моделей
for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    if name in ["SVM", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    end_time = time.time()
    print(f"{name}: Accuracy = {accuracy:.4f}, MSE = {mse:.4f}, Time = {end_time - start_time:.2f} seconds\n")

    # Сохранение модели
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"car_evaluation/{name}_{timestamp}.pkl"
    joblib.dump(model, filename)
