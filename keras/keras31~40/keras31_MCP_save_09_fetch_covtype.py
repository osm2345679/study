# 23-3카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# 1. 내부 연산(Matrix Multiplication 등)에 사용할 스레드 수
tf.config.threading.set_intra_op_parallelism_threads(16)

# 2. 독립적인 연산들을 병렬로 처리할 스레드 수
tf.config.threading.set_inter_op_parallelism_threads(16)

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

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=54))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
path = './_save/keras31/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k31_9_", date, "-", filename])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)

start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=128,
    verbose=1,
    validation_split=0.25,
    callbacks=[es, mcp]
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
#
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 3s 715us/step - acc: 0.8427 - loss: 0.3711
# loss :  0.3711288869380951
# acc :  0.8427
# 걸린 시간 :  1549 2 초
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 436us/step 
# acc :  0.8427
#
# MinMaxScaler
# Epoch 623/3000
# 2724/2724 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - acc: 0.8817 - loss: 0.2923 - val_acc: 0.8773 - val_loss: 0.3020
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 621us/step - acc: 0.8840 - loss: 0.2928
# loss :  0.29275277256965637
# acc :  0.884
# 걸린 시간 :  1793 2 초
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 431us/step 
# acc :  0.884
#
# StandardScaler
# Epoch 622/3000
# 2724/2724 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - acc: 0.8827 - loss: 0.2889 - val_acc: 0.8787 - val_loss: 0.2998
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 3s 850us/step - acc: 0.8828 - loss: 0.2928 
# loss :  0.2928265631198883
# acc :  0.8828
# 걸린 시간 :  2172 2 초
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 563us/step 
# acc :  0.8828
#
# MaxAbsScaler
# Epoch 738/3000
# 2724/2724 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - acc: 0.8775 - loss: 0.2978 - val_acc: 0.8744 - val_loss: 0.3060
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - acc: 0.8783 - loss: 0.2979   
# loss :  0.29791995882987976
# acc :  0.8783
# 걸린 시간 :  2206 2 초
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 565us/step 
# acc :  0.8783
#
# RobustScaler
# Epoch 754/3000
# 2724/2724 ━━━━━━━━━━━━━━━━━━━━ 3s 964us/step - acc: 0.8882 - loss: 0.2742 - val_acc: 0.8837 - val_loss: 0.2848
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 669us/step - acc: 0.8859 - loss: 0.2805
# loss :  0.2804836928844452
# acc :  0.8859
# 걸린 시간 :  2207 2 초
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 428us/step 
# acc :  0.8859