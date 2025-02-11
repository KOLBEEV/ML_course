import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os

# Ограничение количества ядер
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Загрузка датасета Adult
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
    "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Загрузка данных с URL
data = pd.read_csv(url, names=columns, sep=r'\s*,\s*', engine='python')

# Предобработка данных
data = data.replace('?', pd.NA)  # Заменить '?' на NA
data = data.dropna()  # Удалить строки с NA
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)  # Преобразуем целевую переменную в бинарный формат

# Преобразуем категориальные признаки в числовые с помощью OneHotEncoding
data = pd.get_dummies(data, drop_first=True)

# Демонстрация первых 5 строк
print("First 5 rows of the dataset:")
print(data.head())

# Статистическое описание данных
print("\nStatistical description of the dataset:")
print(data.describe())

# Уникальные значения для категориальных признаков
print("\nUnique values in categorical columns:")
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"{column}: {data[column].unique()}")

# Разделение на признаки и целевую переменную
X = data.drop('income', axis=1)
y = data['income']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
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

    # Сохраняем модель
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"adult/{name}_{timestamp}.pkl"
    joblib.dump(model, filename)
