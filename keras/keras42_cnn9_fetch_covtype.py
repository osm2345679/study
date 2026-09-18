# 23-3카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import datetime

# acc = 0.93

#1. 데이터
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']

print(x, y)
# [[2.596e+03 5.100e+01 3.000e+00 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.590e+03 5.600e+01 2.000e+00 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.804e+03 1.390e+02 9.000e+00 ... 0.000e+00 0.000e+00 0.000e+00]
#  ...
#  [2.386e+03 1.590e+02 1.700e+01 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.384e+03 1.700e+02 1.500e+01 ... 0.000e+00 0.000e+00 0.000e+00]
#  [2.383e+03 1.650e+02 1.300e+01 ... 0.000e+00 0.000e+00 0.000e+00]] [5 5 2 ... 3 3 3]
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y), np.unique(y).size)  # [1 2 3 4 5 6 7] 7
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

y = pd.get_dummies(y)
print(y.shape)  # (581012, 7)
# print(y[:10])
#        1      2      3      4      5      6      7
# 0  False  False  False  False   True  False  False
# 1  False  False  False  False   True  False  False
# 2  False   True  False  False  False  False  False
# 3  False   True  False  False  False  False  False
# 4  False  False  False  False   True  False  False
# 5  False   True  False  False  False  False  False
# 6  False  False  False  False   True  False  False
# 7  False  False  False  False   True  False  False
# 8  False  False  False  False   True  False  False
# 9  False  False  False  False   True  False  False

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,6,3,3)
x_test = x_test.reshape(-1,6,3,3)

#2. 모델 구성
model = Sequential()
# model.add(Dense(50, input_dim=54))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(7, activation='softmax'))

model.add(Conv2D(32, (2,1), input_shape=(6,3,3)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,1), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())    # 
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(7, activation='softmax'))  # (10,)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=256,
    verbose=1,
    validation_split=0.25,
    callbacks=[es]
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1], 4))

print("걸린 시간 : ", round(end_time-start_time), 2, "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_argmax)
print("acc : ", round(acc, 4))

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='blue', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label='val_loss')
plt.plot(hist.history['acc'], c='orange', label='acc')
plt.plot(hist.history['val_acc'], c='red', label='val_acc')
plt.legend(loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss / Acc / Val_acc')
plt.grid()
plt.show()

# Results 
# Epoch 276/3000
# 1362/1362 [==============================] - 3s 2ms/step - loss: 0.8120 - acc: 0.6378 - val_loss: 0.7505 - val_acc: 0.6700
# 3632/3632 [==============================] - 4s 1ms/step - loss: 0.7508 - acc: 0.6688
# loss :  0.7507870197296143
# acc :  0.6688
# 걸린 시간 :  783 2 초
# 3632/3632 [==============================] - 3s 759us/step
# acc :  0.6688