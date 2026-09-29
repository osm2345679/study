# 데이터셋 - https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import datetime
import time


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
# print(dataset.shape)    # (420551, 15)
# print(dataset[:3])
#                      p (mbar)  T (degC)  Tpot (K)  Tdew (degC)  rh (%)  VPmax (mbar)  VPact (mbar)  VPdef (mbar)  sh (g/kg)  H2OC (mmol/mol)  rho (g/m**3)  wv (m/s)  max. wv (m/s)  wd (deg)
# Date Time                                                                                                                                                                                    
# 01.01.2009 00:10:00    996.52     -8.02    265.40        -8.90    93.3          3.33          3.11          0.22       1.94             3.12       1307.75      1.03           1.75     152.3
# 01.01.2009 00:20:00    996.57     -8.41    265.01        -9.28    93.4          3.23          3.02          0.21       1.89             3.03       1309.80      0.72           1.50     136.1
# 01.01.2009 00:30:00    996.53     -8.51    264.91        -9.31    93.9          3.21          3.01          0.20       1.88             3.02       1310.24      0.19           0.63     171.6


# Split pred data and y
# arr_dataset = np.array(dataset)
print(dataset.shape)
print(np.array(dataset).shape)   # (144, 14)
print(dataset[0][13])
exit()
y_dataset = arr_dataset[:, -1]
# print(y_dataset.shape)  # (420551,)
# print(y_dataset[:3])    # [152.3 136.1 171.6]

pred_dataset = arr_dataset[-size:]
# print(pred_dataset.shape)   # (144, 14)

arr_dataset = arr_dataset[:-size, :-1]
# print(arr_dataset.shape)    # (420407, 13)


# Split for time series data
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


x = split_x(arr_dataset, size)
# print(x.shape)    # (420264, 144, 13)

y = y_dataset[420551-420264:]
# print(y.shape)  # (420264,)

y = split_x(y, size)
print(y.shape)  # (420121, 144)
exit()

# Data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# print(x_train.shape, y_train.shape) # (336211, 144, 13) (336211,)
# print(x_test.shape, y_test.shape)   # (84053, 144, 13) (84053,)


# Scaling
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train, x_train)
# x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(LSTM(units=100, input_shape=(144, 13), return_sequences=True))    # (N, 144, 13) -> (144, 13)
model.add(LSTM(units=200, return_sequences=True))
model.add(LSTM(units=200, return_sequences=True))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)
path = './_save/keras58/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k58_1_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.5
)
learning_rate = 0.01

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3000,
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
y_pred = model.predict(x_test)
# print(y_pred.shape) # (84053, 1)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_pred)
print("mse : ", mse)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_pred)
print("rmse : ", rmse)


# Predict pred_dataset
pred_dataset = split_x(pred_dataset, size)
y_pred = model.predict(pred_dataset)
print("2016-12-31 00:10 ~ 2017-1-1 00:00의 예측값 : ", y_pred)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# 제출
y_pred.to_csv(path + "submit/" + "submit_" + date + ".csv")

# Results
# 985/985 [==============================] - 16s 14ms/step - loss: 7820.7085 - val_loss: 7521.9639 - lr: 0.0100
# 2026-09-29 16:35:02.675892: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 2517547968 exceeds 10% of free system memory.
# 10507/10507 [==============================] - 41s 4ms/step - loss: 7516.0283
# loss :  7516.0283203125
# 2627/2627 [==============================] - 10s 3ms/step
# r2 :  -0.0008309071286696224
# mse :  7526.704580314724
# rmse :  86.75658234574898
# 1/1 [==============================] - 0s 192ms/step
# 2016-12-31 00:10 ~ 2017-1-1 00:00의 예측값 :  [[126.318634]]
# 걸린 시간 :  16.7 초
