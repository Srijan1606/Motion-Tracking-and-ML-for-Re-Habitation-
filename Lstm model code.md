import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
dataset = pd.read_csv(r"C:\Users\hp\sensor_data_clustered.csv")
features = dataset[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']].values
labels = dataset['Cluster'].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
def create_sequences(data, labels, seq_length=30):
    sequences = []
    seq_labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        seq_labels.append(labels[i+seq_length])
    return np.array(sequences), np.array(seq_labels)
seq_length = 30
X, y = create_sequences(features_scaled, labels_encoded, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))  
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
model.save(r'C:\Users\hp\sensor_model.h5')
print("Model saved as sensor_model.h5")
