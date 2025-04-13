import matplotlib.pyplot as plt

training_times = {
    'Random Forest': 0.74,
    'XGBoost': 0.17,
    'MLP': 8.09,
    'CNN': 4.57,
    'RNN (LSTM)': 8.51
}

plt.figure(figsize=(10, 5))
plt.bar(training_times.keys(), training_times.values(), color='skyblue')
plt.ylabel('Секунды')
plt.title('Время обучения моделей')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()