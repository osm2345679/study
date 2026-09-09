from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data()

print(x_train.shape())
print(y_train.shape())
print(x_test.shape())
print(y_test.shape())

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=np.size(y, axis=1), activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)
hist = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=32,
    verbose=1,
    callbacks=[es],
    validation_split=0.25
)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", acc_score)

plt.figure(figsize=(9,6))
plt.plot(hist.history['acc'], c='red', label='accuracy')
plt.plot(hist.history['val_acc'], c='green', label='val_accuracy')
plt.plot(hist.history['loss'], c='blue', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label='val_loss')
plt.legend(loc='')