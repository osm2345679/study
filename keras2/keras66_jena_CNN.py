from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
size = 144
path = './_data/kaggle_jena/'

dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

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


x_train = x_train.reshape(-1, 12, 12, 13)
x_test = x_test.reshape(-1, 12, 12, 13)
# print(x_train.shape, x_test.shape)  # (336096, 144, 13) (84024, 144, 13)

#2. 모델 구성
# model = Sequential()
# model.add(LSTM(units=100, input_shape=(144, 13), dropout=0.2))    # (N, 144, 13) -> (144, 13)
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(144))

model = Sequential()
model.add(Conv2D(100, (5,5), input_shape=(12, 12, 13)))
model.add(Conv2D(100, (5,5), activation='relu'))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(144))

#3. 컴파일, 훈련
learning_rate = 0.001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=20,
    restore_best_weights=True
)

filepath = './_save/keras66/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([filepath, "keras66", date, "-", filename])

mcp = ModelCheckpoint(
    filepath = filepath,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=20,
    factor=0.5
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=512,
    verbose=1,
    validation_split=0.2,
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
pred_dataset = pred_dataset.reshape(-1, 12, 12, 13)

y_pred = model.predict(pred_dataset)
# print("2016-12-31 00:10 ~ 2017-1-1 00:00의 예측값 : ", y_pred)
# print(y_pred.shape) # (1, 144)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 119: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 526/526 [==============================] - 5s 9ms/step - loss: 0.9854 - val_loss: 1.0926 - lr: 0.0010
# Epoch 119: early stopping
# 2026-10-06 11:12:04.813743: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2516686848 exceeds 10% of free system memory.
# 10503/10503 [==============================] - 14s 1ms/step - loss: 0.8979
# loss :  0.8979353904724121
# 2626/2626 [==============================] - 3s 1ms/step
# r2 :  0.9857148670493001
# mse :  1.0084638766673573
# rmse :  1.0042230213788954
# 1/1 [==============================] - 0s 44ms/step
# 걸린 시간 :  558.35 초