import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def build_model():
    model = Sequential([
        LSTM(units=64, input_shape=(20,4)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model



