#23-2 카피
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
import numpy as np
import datetime
import time

# acc = 0.95

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)
print(x, y)

y = pd.get_dummies(y)
print(y)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=52,
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
model.add(Dense(10, input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
path = './_save/keras31/'
model = load_model(path + 'k31_8_0914-1444-0014-0.0365.keras')

# path = './_save/keras31/'
# date = datetime.datetime.now()
# date = date.strftime("%m%d-%H%M")
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k31_8_", date, "-", filename])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=100,
#     restore_best_weights=True
# )
# mcp = ModelCheckpoint(
#     monitor = 'val_loss',
#     mode = 'auto',
#     verbose = 0,
#     save_best_only = True,
#     filepath = filepath
# )
# start_time = time.time()
# model.fit(
#     x_train, y_train,
#     epochs=3000,
#     batch_size=8,
#     verbose=1,
#     callbacks=[es, mcp],
#     validation_split=0.25
# )
# end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1], 2))

# print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_argmax)
print("acc : ", round(acc, 4))

# Results
# 
# Epoch 357/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 0.9528 - loss: 0.1246 - val_acc: 0.9444 - val_loss: 0.1103
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 0.9722 - loss: 0.1414 
# loss :  0.14142635464668274
# acc :  0.97
# 걸린 시간 :  24.74 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step
# acc :  0.9722
#
# MinMaxScaler
# Epoch 125/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 1.0000 - loss: 6.7710e-04 - val_acc: 0.9444 - val_loss: 0.0777
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - acc: 1.0000 - loss: 0.0493
# loss :  0.049326471984386444
# acc :  1.0
# 걸린 시간 :  9.47 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
# acc :  1.0
#
# StandardScaler
# Epoch 115/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 1.0000 - loss: 1.4544e-04 - val_acc: 0.9722 - val_loss: 0.1946
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - acc: 1.0000 - loss: 0.0536
# loss :  0.05360769107937813
# acc :  1.0
# 걸린 시간 :  8.53 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
# acc :  1.0
#
# MaxAbsScaler
# Epoch 251/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - acc: 1.0000 - loss: 0.0021 - val_acc: 0.9722 - val_loss: 0.0343
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - acc: 0.9722 - loss: 0.0797  
# loss :  0.07973544299602509
# acc :  0.97
# 걸린 시간 :  19.35 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
# acc :  0.9722
#
# RobustScaler
# Epoch 134/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 1.0000 - loss: 9.7202e-05 - val_acc: 0.9722 - val_loss: 0.0422
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - acc: 1.0000 - loss: 0.0331
# loss :  0.0331292599439621
# acc :  1.0
# 걸린 시간 :  9.78 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
# acc :  1.0