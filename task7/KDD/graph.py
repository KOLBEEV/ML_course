import matplotlib.pyplot as plt

training_times = {
    'Random Forest': 9.27,
    'XGBoost': 0.93,
    'MLP': 32.75,
    'CNN': 36.25,
    'RNN (LSTM)': 276.59
}

plt.figure(figsize=(10, 5))
plt.bar(training_times.keys(), training_times.values())
plt.ylabel('Секунды')
plt.title('Время обучения моделей')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()