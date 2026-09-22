# 40-1 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scaling
x_train = x_train/255
x_test = x_test/255

# one-hot encoding
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
# exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
# model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # (20,20,64). 파라미터 이름도 적어봄
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())   # (10,10,64)
# model.add(Conv2D(32, (3,3), activation='relu')) # (8,8,32)
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling2D())    # 
# # 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
# model.add(Dense(10, activation='softmax'))  # (10,)

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=64,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()

#4. 평가, 예측
print("=========model.evaluate==========")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)

print("accuracy_score : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 72/2000
# 375/375 [==============================] - 1s 2ms/step - loss: 0.0242 - acc: 0.9930 - val_loss: 0.1019 - val_acc: 0.9803
# =========model.evaluate==========
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0850 - acc: 0.9806
# loss :  0.08503751456737518
# acc :  0.9805999994277954
# 313/313 [==============================] - 0s 649us/step
# accuracy_score :  0.9806
# 걸린 시간 :  49.28 초