import os
import warnings
import pandas as pd
import numpy as np
import time
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Подавление предупреждений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Загрузка и подготовка данных
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Очистка текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Векторизация текста для классических моделей
vectorizer = TfidfVectorizer(max_features=3000)
X_vect = vectorizer.fit_transform(df['text']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Классические модели
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
}

metrics = {}
times = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    elapsed = end - start
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

print("Метрики моделей:")
for model_name, m in metrics.items():
    print(f"{model_name}: {m}")
print("\nВремя обучения (в секундах):", times)

# Подготовка текста для нейросетей
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=max_len)

X_train_nn, X_test_nn, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# CNN
X_train_cnn = np.expand_dims(X_train_nn, axis=2)
X_test_cnn = np.expand_dims(X_test_nn, axis=2)

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
end = time.time()
print("Время обучения CNN:", round(end - start, 2), "сек")

# RNN (LSTM)
rnn_model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
start = time.time()
rnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=64, validation_split=0.2)
end = time.time()
print("Время обучения RNN (LSTM):", round(end - start, 2), "сек")
