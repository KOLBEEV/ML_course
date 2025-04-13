import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Подавление ненужных предупреждений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключить сообщения TensorFlow
warnings.filterwarnings('ignore')          # Отключить предупреждения от scikit-learn/xgboost

# Загрузка данных
column_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
                'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
                'root_shell','su_attempted','num_root','num_file_creations','num_shells',
                'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
                'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
                'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
                'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
                'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

df = pd.read_csv('KDDTrain+.txt', names=column_names, header=None)

df.drop('difficulty', axis=1, inplace=True)
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
df['label'] = np.where(df['label'] == 'normal', 0, 1)

X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300, random_state=42)
}

metrics = {}
times = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    elapsed = end - start
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    times[name] = elapsed

# Вывод метрик и времени
print("Метрики моделей:", metrics)
print("Время обучения (в секундах):", times)

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
end = time.time()
print("Время обучения CNN:", end - start, "сек")

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
print("Время обучения RNN (LSTM):", end - start, "сек")
