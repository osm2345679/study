from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

# scale
x_train = x_train/255
x_test = x_test/255

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

# one-hot encoding
ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 183/2000
# 313/313 [==============================] - 7s 21ms/step - loss: 0.4126 - acc: 0.8580 - val_loss: 0.7469 - val_acc: 0.7785
# 313/313 [==============================] - 1s 2ms/step - loss: 0.7127 - acc: 0.7769
# loss :  0.7126936912536621
# acc :  0.7768999934196472
# 313/313 [==============================] - 1s 2ms/step
# acc :  0.7769
# 걸린 시간 :  1213.53 초