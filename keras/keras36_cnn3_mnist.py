# 36-2 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000,)
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test)) # 255 0

###### 스케일링 1
x_train = x_train/255.  # python 기초 - . 붙여서 실수형으로 변환 (. 붙이면 .0으로 판단하고 실수로 처리함)
x_test = x_test/255   # python3부터는 정수 나눗셈하면 기본적으로 실수 나온다고 함
# 이미지 데이터인 거 아니까 그냥 255로 나누기

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test)) # 1.0 0.0

###### 스케일링 2
# x_train = (x_train-127.5)/127.5 # -1~1로 스케일링. 나눈 다음 그 값에서 1빼는 식으로 해도 됨
# x_test = (x_test-127.5)/127.5

# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test)) # 1.0 -1.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape)    # (60000, 28, 28, 1)    # 4차원 형식되게 reshape
print(x_test.shape) # (10000, 28, 28, 1)

# exit()

######## One-hot Encoding ######
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_train = encoder.fit_transform(y_train)

y_test = y_test.reshape(-1,1)
y_test = encoder.fit_transform(y_test)

print(y_train.shape)    # (60000,10)
print(y_test.shape) # (10000,10)

#2. 모델 구성
model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))   # (26, 26, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # (24,24,32). 파라미터 이름도 적어봄
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation='relu')) # (23,23,32)
# model.add(Conv2D(16, (2,2), activation='relu')) # (22,22,16)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu')) # (21,21,16)
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu')) # (20,20,16)
# model.add(Flatten())    # (6400,)
# # 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
# model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
# model.add(Dropout(0.2))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(10, activation='softmax'))  # (10,)

model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # (20,20,64). 파라미터 이름도 적어봄
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu')) # (16,16,32)
model.add(Conv2D(32, (5,5), activation='relu')) # (12,12,32)
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu')) # (4,4,32)
model.add(Dropout(0.2))
model.add(Flatten())    # (512,)
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax'))  # (10,)

model.summary()

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
    batch_size=128,
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
# print(y_pred[:20])
# print(y_pred.shape)

# print(y_test[:20])
# print(y_test.shape)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# print(y_pred[:20])
# print(y_pred.shape)

# print(y_test[:20])
# print(y_test.shape)

acc = accuracy_score(y_test, y_pred)

print("accuracy_score : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results

# CPU
# Epoch 50/50
# 375/375 ━━━━━━━━━━━━━━━━━━━━ 11s 28ms/step - acc: 0.9913 - loss: 0.0259 - val_acc: 0.9888 - val_loss: 0.0567
# =========model.evaluate==========
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - acc: 0.9901 - loss: 0.0380     
# loss :  0.03803085535764694
# acc :  0.9901000261306763
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
# acc_score :  0.9901
# 걸린 시간 :  561.38

# GPU
# Epoch 50/50
# 375/375 [==============================] - 2s 7ms/step - loss: 0.0229 - acc: 0.9927 - val_loss: 0.0532 - val_acc: 0.9902
# =========model.evaluate==========
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0467 - acc: 0.9900
# loss :  0.04671752452850342
# acc :  0.9900000095367432
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.99
# 걸린 시간 :  120.55 초

# parameter tuning
# Epoch 56/2000
# 375/375 [==============================] - 3s 9ms/step - loss: 0.0214 - acc: 0.9951 - val_loss: 0.0598 - val_acc: 0.9914
# =========model.evaluate==========
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0274 - acc: 0.9940 
# loss :  0.027361255139112473
# acc :  0.9940000176429749
# 313/313 [==============================] - 0s 1ms/step
# accuracy_score :  0.994
# 걸린 시간 :  192.05 초