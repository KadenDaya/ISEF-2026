import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from model.model import build_model
from utils import encode_batch, scale_output

df = pd.read_csv('data/train.csv')

x = encode_batch(df['gRNA_sequence'])
y = scale_output(df['UV_score'].values)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = build_model()
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

model.save('model/model.h5')
