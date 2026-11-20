# 36-3 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

###### 스케일링 1
x_train = x_train/255.  # python 기초 - . 붙여서 실수형으로 변환 (. 붙이면 .0으로 판단하고 실수로 처리함)
x_test = x_test/255   # python3부터는 정수 나눗셈하면 기본적으로 실수 나온다고 함

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

######## One-hot Encoding ######
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_train = encoder.fit_transform(y_train)

y_test = y_test.reshape(-1,1)
y_test = encoder.fit_transform(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # (20,20,64). 파라미터 이름도 적어봄
# model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())    # (512,)
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax'))  # (10,)

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

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)

print("accuracy_score : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results - BatchNorm 안 썼을 때
# Epoch 44/2000
# 375/375 [==============================] - 4s 10ms/step - loss: 0.0124 - acc: 0.9961 - val_loss: 0.0879 - val_acc: 0.9884
# =========model.evaluate==========
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0506 - acc: 0.9856 
# loss :  0.050550226122140884
# acc :  0.9855999946594238
# 313/313 [==============================] - 2s 5ms/step
# accuracy_score :  0.9856
# 걸린 시간 :  165.14 초

# Results - BatchNormalization
# Epoch 48/2000
# 375/375 [==============================] - 3s 9ms/step - loss: 0.0096 - acc: 0.9970 - val_loss: 0.0960 - val_acc: 0.9877
# =========model.evaluate==========
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0490 - acc: 0.9875
# loss :  0.04897410795092583
# acc :  0.987500011920929
# 313/313 [==============================] - 1s 1ms/step
# accuracy_score :  0.9875
# 걸린 시간 :  184.81 초