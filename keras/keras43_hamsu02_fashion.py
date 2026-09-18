# 39-2 카피
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)

# plt.imshow(x_train[0])
# plt.show()

# scale
x_train = x_train / 255
x_test = x_test / 255

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# one-hot encoding
ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델 구성
# model = Sequential()
# model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # 24,24,64
# model.add(Conv2D(64, (5,5), activation='relu')) # 20, 20, 64
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())   # 10, 10, 64
# model.add(Conv2D(32, (3,3), activation='relu')) # 8,8,32
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu')) # 6,6,32
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(28,28,1))
conv1 = Conv2D(64, (5,5))(input1)
conv2 = Conv2D(64, (5,5), activation='relu')(conv1)
drop1 = Dropout(0.2)(conv2)
pool1 = MaxPooling2D()(drop1)
conv3 = Conv2D(32, (3,3), activation='relu')(pool1)
drop2 = Dropout(0.2)(conv3)
conv4 = Conv2D(32, (3,3), activation='relu')(drop2)
drop3 = Dropout(0.2)(conv4)
gap1 = GlobalAveragePooling2D()(drop3)
fc1 = Dense(32, activation='relu')(gap1)
drop4 = Dropout(0.2)(fc1)
fc2 = Dense(16, activation='relu')(drop4)
output1 = Dense(10, activation='softmax')(fc2)
model = Model(inputs=input1, outputs=output1)

model.summary()
#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
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
# Epoch 105/2000
# 375/375 [==============================] - 3s 7ms/step - loss: 0.1832 - acc: 0.9333 - val_loss: 0.2419 - val_acc: 0.9137
# 313/313 [==============================] - 1s 2ms/step - loss: 0.2493 - acc: 0.9141
# loss :  0.24933584034442902
# acc :  0.9140999913215637
# 313/313 [==============================] - 0s 1ms/step
# acc :  0.9141
# 걸린 시간 :  264.45 초