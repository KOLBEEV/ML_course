import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input

warnings.filterwarnings('ignore')

# Загрузка данных
file_path = "dataset.csv"  # или путь к вашему CSV
phishing_df = pd.read_csv(file_path)

# Удаление index если есть
if 'index' in phishing_df.columns:
    phishing_df.drop(columns=['index'], inplace=True)

# Приведение целевой переменной к значениям 0 и 1
phishing_df['Result'] = phishing_df['Result'].replace(-1, 0)

# Целевая переменная
X = phishing_df.drop('Result', axis=1)
y = phishing_df['Result']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модели
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
}

metrics = {}
times = {}

# Обучение моделей
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    times[name] = elapsed

print("Метрики моделей:", metrics)
print("Время обучения:", times)

# CNN и RNN
X_train_cnn = np.expand_dims(X_train.values.astype('float32'), axis=2)
X_test_cnn = np.expand_dims(X_test.values.astype('float32'), axis=2)

# CNN
cnn_model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
start = time.time()
cnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=64, validation_split=0.2)
print("Время обучения CNN:", time.time() - start)

# RNN
rnn_model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
start = time.time()
rnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=64, validation_split=0.2)
print("Время обучения RNN:", time.time() - start)