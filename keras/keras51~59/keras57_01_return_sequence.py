# 55-2에서 데이터만 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import datetime


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])    # 목표 80

size = 3

print(x.shape, y.shape) # (13, 3) (13,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-size+1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

# split_x로 전처리
# x = x.reshape(-1, 3, 1)
x = np.expand_dims(x, axis=2)
# print(x.shape)  # (13, 3, 1)

#2. 모델 구성
# model = Sequential()
# model.add(LSTM(units=100, input_shape=(3, 1), return_sequences=True))
# model.add(LSTM(units=100, return_sequences=True))
# model.add(LSTM(units=100, return_sequences=True))
# model.add(LSTM(units=100, return_sequences=True))
# model.add(LSTM(units=100))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1), return_sequences=True))
model.add(LSTM(units=10, return_sequences=True))
model.add(LSTM(units=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    verbose=1,
    restore_best_weights=True
)
path = './_save/keras57/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k57_1_", date, "-", filename])

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
    patience=100,
    verbose=1,
    factor=0.5
)
learning_rate = 0.01

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.fit(
    x, y,
    epochs=5000,
    batch_size=32,
    verbose=1,
    validation_split=0.01,
    callbacks=[es, mcp, rlr]
)

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)


x_predict = x_predict.reshape(-1,3,1)
# print(x_predict)
# print(x_predict.shape)  # (3, 5, 2)
y_pred = model.predict(x_predict)

print("[50,60,70]의 예측값 : ", y_pred)

# Results
# Epoch 59/3000
#  1/12 [=>............................] - ETA: 0s - loss: 2.8600Restoring model weights from the end of the best epoch: 9.
# 12/12 [==============================] - 0s 3ms/step - loss: 20.6372 - val_loss: 118.6009 - lr: 0.0025
# Epoch 59: early stopping
# 1/1 [==============================] - 0s 86ms/step - loss: 12.2225
# loss :  12.222495079040527
# 1/1 [==============================] - 0s 85ms/step
# [50,60,70]의 예측값 [[78.24288]]

# LSTM 여러 번
# Epoch 117: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.
# 1/1 [==============================] - 0s 18ms/step - loss: 0.0053 - val_loss: 56.2891 - lr: 0.0100
# Epoch 117: early stopping
# 1/1 [==============================] - 0s 12ms/step - loss: 2.9860
# loss :  2.9860177040100098
# 1/1 [==============================] - 0s 464ms/step
# [50,60,70]의 예측값 :  [[71.76533]]

# Epoch 136: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.
# 1/1 [==============================] - 0s 41ms/step - loss: 0.1833 - val_loss: 24.2380 - lr: 0.0100
# Epoch 136: early stopping
# 1/1 [==============================] - 0s 16ms/step - loss: 43.7915
# loss :  43.79152297973633
# 1/1 [==============================] - 1s 520ms/step
# [50,60,70]의 예측값 :  [[78.66775]]