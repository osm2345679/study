from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
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
model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

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
# Epoch 113/2000
# 375/375 [==============================] - 4s 10ms/step - loss: 0.1459 - acc: 0.9464 - val_loss: 0.2746 - val_acc: 0.9109
# 313/313 [==============================] - 1s 2ms/step - loss: 0.2683 - acc: 0.9107
# loss :  0.26830995082855225
# acc :  0.9107000231742859
# 313/313 [==============================] - 0s 1ms/step
# acc :  0.9107
# 걸린 시간 :  456.01 초