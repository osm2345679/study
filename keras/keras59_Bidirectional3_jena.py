# 58-1 카피
# 데이터셋 - https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import datetime
import time
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#1. 데이터

# 2016-12-31 00:10 ~ 2017-1-1 00:00 맞추기. 총 144개. timestep 144
# 조건
# 맞춰야하는 열 데이터는 기존 데이터와 분리
# y는 15열인 wd
# x = (N, 144, 13), y = (N, 144, 1)

size = 144


# Data Load
path = './_data/kaggle_jena/'
dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(type(dataset))    # <class 'pandas.core.frame.DataFrame'>
# print(dataset.shape)    # (420551, 14)
# print(dataset[:3])
#                      p (mbar)  T (degC)  Tpot (K)  Tdew (degC)  rh (%)  VPmax (mbar)  VPact (mbar)  VPdef (mbar)  sh (g/kg)  H2OC (mmol/mol)  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
# Date Time                                                                                                                                                                                    
# 01.01.2009 00:10:00    996.52     -8.02    265.40        -8.90    93.3          3.33          3.11          0.22       1.94             3.12       1307.75      1.03           1.75     152.3
# 01.01.2009 00:20:00    996.57     -8.41    265.01        -9.28    93.4          3.23          3.02          0.21       1.89             3.03       1309.80      0.72           1.50     136.1
# 01.01.2009 00:30:00    996.53     -8.51    264.91        -9.31    93.9          3.21          3.01          0.20       1.88             3.02       1310.24      0.19           0.63     171.6

'''
보다 깔끔한 코드 찾고 살짝 수정해서 정리함
찾아보니 경우에 따라 .columns보단 .iloc[]가 깔끔할듯
찾아보니 pandas에서도 iloc, loc는 다차원 슬라이싱 가능하다고 함.
ㄴ 자체 객체의 __getitem__을 오버라이드 했다고 함
ㄴ iloc : Integer location

1안 - x-y 순서대로
x = dataset[:-2*size].drop(dataset.columns[1], axis=1)
print(x.shape)  # (420263, 13)

pred_dataset = dataset[-size:].drop(dataset.columns[1], axis=1)
print(pred_dataset.shape)   # (144, 13)

y = dataset[size:-size][dataset.columns[1]]
print(y.shape)  # (420263,)


2안 - drop 하고 x 자름
y = dataset[size:-size][dataset.columns[1]]
print(y.shape)  # (420263,)

dataset = dataset.drop(dataset.columns[1], axis=1)  # drop y

x = dataset[:-2*size]
print(x.shape)  # (420263, 13)

pred_dataset = dataset[-size:]
print(pred_dataset.shape)   # (144, 13)

'''

# Split pred data and y
arr_dataset = np.array(dataset)
y_dataset = arr_dataset[:, 1]
# print(y_dataset.shape)  # (420551,)
# print(y_dataset[:3])    # [-8.02 -8.41 -8.51]

dataset = dataset.drop(['T (degC)'], axis=1)  # drop y

arr_dataset = np.array(dataset)
pred_dataset = arr_dataset[-size:]
# print(pred_dataset.shape)   # (144, 13)

arr_dataset = arr_dataset[:-size]
# print(arr_dataset.shape)    # (420407, 13)

# Split for time series data
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

# 사용할 x 먼저 정리
# 현재 x를 그냥 split하면 마지막 144행은 해당하는 y가 없으므로(정확히는 잘라내서 못 쓰는 상태)
# 마지막 144을 자르고(마지막 144행까지만 선택하고)
x = arr_dataset[:-size] # arr_dataset[:-144]

# split함
x = split_x(x, size)

# print(x.shape)    # (420120, 144, 13)

# y의 경우 지금 원데이터셋에서 1열만 선택한 상태인데
# 여기서 첫 144행까지는 해당하는 x가 없고 마지막 144행은 잘라내야 하므로
y = y_dataset[size:-size]   # y_dataset[144:-144]
# print(y.shape)  # (420263,)

# 그 이후 split함
y = split_x(y, size)
# print(y.shape)  # (420120, 144)
# exit()

# Data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# print(x_train.shape, y_train.shape) # (336096, 144, 13) (336096, 144)
# print(x_test.shape, y_test.shape)   # (84024, 144, 13) (84024, 144)
# exit()

# Scaling
# sklearn의 scaler들이 2차원 배열을 입력받아서 현재 데이터는 3차원이라 그대로는 못 넣음
# reshape
x_train = x_train.reshape(-1, 13)
x_test = x_test.reshape(-1,13)
# print(x_train.shape, x_test.shape)  # (48397824, 13) (12099456, 13)

# scaling
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, x_train)
x_test = scaler.transform(x_test)

# scaling 했으니 다시 3차원으로 reshape
x_train = x_train.reshape(-1, 144, 13)
x_test = x_test.reshape(-1, 144, 13)
# print(x_train.shape, x_test.shape)  # (336096, 144, 13) (84024, 144, 13)

# exit()

#2. 모델 구성
# model = Sequential()
# model.add(LSTM(units=100, input_shape=(144, 13)))    # (N, 144, 13) -> (144, 13)
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(144))

model = Sequential()
model.add(Bidirectional(LSTM(units=100), input_shape=(144, 13)))    # (N, 144, 13) -> (144, 13)
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(144))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    verbose=1,
    restore_best_weights=True
)
filepath = './_save/keras59/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([filepath, "k59_1_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=5,
    verbose=1,
    factor=0.5
)
learning_rate = 0.01

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=512,
    verbose=1,
    validation_split=0.25,
    callbacks=[es, mcp, rlr]
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_train, y_train)
print("loss : ", results)


# Predict scores
# print(x_test.shape)
y_pred = model.predict(x_test)
# print(y_pred.shape) # (84024, 144)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_pred)
print("mse : ", mse)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("rmse : ", rmse)


# Predict pred_dataset
# print(pred_dataset.shape)   # (144, 13)

pred_dataset = split_x(pred_dataset, size)
# print(pred_dataset.shape)   # (1, 144, 13)

y_pred = model.predict(pred_dataset)
# print("2016-12-31 00:10 ~ 2017-1-1 00:00의 예측값 : ", y_pred)
# print(y_pred.shape) # (1, 144)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# 제출
# to_csv() 함수 사용을 위해 예측결과를 pandas 데이터로 변환
y_pred = y_pred.reshape(-1) # (1, 144)라서 오류나서 (144,)로 변환
pd_pred = pd.DataFrame(y_pred, columns=['T (degC)'])

pd_pred.to_csv(path + "submit/" + "submit_" + date + ".csv")

# print(pred_dataset[:,0].shape, pred_dataset[:,-1].shape)    # (144,) (144,)
# pd_pred = pd.DataFrame(data=(pred_dataset[:,0], pred_dataset[:,-1]), columns=['Date Time', 'wd (deg)'])

# pd_pred.to_csv(path + "submit/" + "submit_.csv")

# exit()

# Results
# Epoch 38: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.
# 493/493 [==============================] - 7s 15ms/step - loss: 1.0726 - val_loss: 1.1065 - lr: 0.0100
# Epoch 38: early stopping
# 2026-09-30 15:56:14.911539: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2516686848 exceeds 10% of free system memory.
# 10503/10503 [==============================] - 38s 4ms/step - loss: 1.0147
# loss :  1.0146934986114502
# 2626/2626 [==============================] - 9s 3ms/step
# r2 :  0.9848753726334883
# mse :  1.0676983606939858
# rmse :  1.0332949049975935
# 1/1 [==============================] - 0s 8ms/step
# 걸린 시간 :  276.67 초

# Dropout + Bidirectional
# Epoch 18: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.
# 493/493 [==============================] - 13s 27ms/step - loss: 1.5112 - val_loss: 2.9754 - lr: 0.0100
# Epoch 18: early stopping
# 2026-09-30 16:54:23.555268: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2516686848 exceeds 10% of free system memory.
# 10503/10503 [==============================] - 67s 6ms/step - loss: 2.6004
# loss :  2.6004228591918945
# 2626/2626 [==============================] - 16s 6ms/step
# r2 :  0.9624740417009323
# mse :  2.649262271772393
# rmse :  1.6276554524138065
# 1/1 [==============================] - 0s 14ms/step
# 걸린 시간 :  246.28 초

# Bidirectional
# Epoch 43: early stopping
# 2026-09-30 17:31:56.264978: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2516686848 exceeds 10% of free system memory.
# 10503/10503 [==============================] - 70s 7ms/step - loss: 0.6933
# loss :  0.6933417320251465
# 2626/2626 [==============================] - 17s 6ms/step
# r2 :  0.9896972474299316
# mse :  0.727304727505039
# rmse :  0.8528216270153092
# 1/1 [==============================] - 0s 17ms/step
# 걸린 시간 :  588.13 초