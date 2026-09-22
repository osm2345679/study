# 39-3 카피
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
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
# model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(32,32,3))) # 30, 30, 64
# model.add(Conv2D(64, (3,3), activation='relu')) # 28,28,64
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())   # 14,14,64
# model.add(Conv2D(64, (3,3), activation='relu')) # 12,12,64
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu')) # 10,10,32
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())   # 5,5,32
# model.add(Conv2D(16, (3,3), activation='relu')) # 3,3,16
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(32,32,3))
conv1 = Conv2D(64, (3,3))(input1)
conv2 = Conv2D(64, (3,3), activation='relu')(conv1)
drop1 = Dropout(0.2)(conv2)
pool1 = MaxPooling2D()(drop1)
conv3 = Conv2D(64, (3,3), activation='relu')(pool1)
drop2 = Dropout(0.2)(conv3)
conv4 = Conv2D(32, (3,3), activation='relu')(drop2)
drop3 = Dropout(0.2)(conv4)
pool2 = MaxPooling2D()(drop3)
conv5 = Conv2D(16, (3,3), activation='relu')(pool2)
drop4 = Dropout(0.2)(conv5)
gap1 = GlobalAveragePooling2D()(drop2)
fc1 = Dense(32, activation='relu')(gap1)
drop5 = Dropout(0.2)(fc1)
fc2 = Dense(16, activation='relu')(drop5)
output1 = Dense(10, activation='softmax')(fc2)
model = Model(inputs=input1, outputs=output1)

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
# Epoch 180/2000
# 313/313 [==============================] - 3s 9ms/step - loss: 0.5714 - acc: 0.8008 - val_loss: 0.6990 - val_acc: 0.7582
# 313/313 [==============================] - 1s 2ms/step - loss: 0.6958 - acc: 0.7643
# loss :  0.6958246231079102
# acc :  0.7642999887466431
# 313/313 [==============================] - 0s 933us/step
# acc :  0.7643
# 걸린 시간 :  549.63 초