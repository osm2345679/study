from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

# plt.imshow(x_train[0])
# plt.show()

# scale
x_train = x_train/255
x_test = x_test/255

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

print(np.unique(y_train, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))

# one-hot encoding
ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)
start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=256,
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

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label='val_loss')
plt.plot(hist.history['acc'], c='blue', label='accuracy')
plt.plot(hist.history['val_acc'], c='red', label='val_accuracy')
plt.legend(loc='upper right')
plt.title('cifar100 loss, acc')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss / Acc / Val_acc')
plt.grid()
plt.show()

# Results

# Epoch 289/2000
# 157/157 [==============================] - 7s 45ms/step - loss: 2.4042 - acc: 0.3455 - val_loss: 2.7588 - val_acc: 0.3183
# 313/313 [==============================] - 1s 2ms/step - loss: 2.6101 - acc: 0.3310
# loss :  2.610139846801758
# acc :  0.3310000002384186
# 313/313 [==============================] - 1s 2ms/step
# acc :  0.331
# 걸린 시간 :  2180.7 초