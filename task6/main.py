import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# Загрузка данных
def load_data():
    train_data = pd.read_csv("KDDTrain+.txt", header=None)
    test_data = pd.read_csv("KDDTest+.txt", header=None)
    return train_data, test_data


# Предобработка данных
def preprocess_data(train_data, test_data):
    # Разделение данных на признаки и целевую переменную
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    # Приведение имен столбцов к строковому типу
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Определение категориальных признаков (1, 2, 3 - категориальные)
    categorical_columns = ["1", "2", "3"]

    # Кодирование категориальных признаков через one-hot encoding
    X_train = pd.get_dummies(X_train, columns=categorical_columns, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=int)

    # Выравнивание наборов данных (если есть несовпадающие признаки, заполняем их нулями)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Преобразование данных в числовой формат
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Удаление или замена NaN значений
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Нормализация числовых признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Кодирование меток классов
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test, label_encoder.classes_


# Создание модели нейронной сети
def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Основной процесс
train_data, test_data = load_data()
X_train, y_train, X_test, y_test, class_names = preprocess_data(train_data, test_data)
model = build_model(X_train.shape[1], len(class_names))

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)


# Визуализация процесса обучения
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


plot_metrics(history)

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_test, y_pred, target_names=class_names))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
