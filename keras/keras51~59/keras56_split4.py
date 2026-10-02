# 56-3 에서 #1. 데이터 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import datetime

#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))    # 96~105로 101~106을 예측.

size = 6

# 데이터를 reshape한 후, split_x 함수로 시계열 데이터로 변환
# (N, 10, 1) -> (N, 5, 2)
# 결과 내보기

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

a = a.reshape(-1, 2)
# print(a.shape)  # (50, 2)

bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)  # (45, 6, 2)

x = bbb[:, :-1] # bbb[0:45, 0:5, 0:1]
y = bbb[:, -1, -1]  # bbb[0:45, 5, 1]
# print(x, y)
# print(x.shape, y.shape) # (45, 5, 2) (45,)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=100, input_shape=(5,2)))  # x.shape가 (45, 5, 2)인 데이터이므로. input_shape : (45, 5, 2) -> (5, 2)
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=300,
    verbose=1,
    restore_best_weights=True
)
path = './_save/keras56/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k56_4_", date, "-", filename])

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
    patience=300,
    verbose=1,
    factor=0.5
)
learning_rate = 0.001

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

x_predict = split_x(x_predict, 5)
# print(x_predict)
# print(x_predict.shape)  # (6, 5)

x_predict = x_predict.reshape(-1,5,2)
# print(x_predict)
# print(x_predict.shape)  # (3, 5, 2)
y_pred = model.predict(x_predict)

print("[96~101]의 예측값 : ", y_pred)

# Results
# Epoch 988/5000
# 1/2 [==============>...............] - ETA: 0s - loss: 7.9704e-05Restoring model weights from the end of the best epoch: 688.
# 2/2 [==============================] - 0s 25ms/step - loss: 1.0683e-04 - val_loss: 5.1899e-04 - lr: 1.2500e-04
# Epoch 988: early stopping
# 2/2 [==============================] - 0s 0s/step - loss: 3.4317e-04
# loss :  0.00034316533128730953
# 1/1 [==============================] - 0s 89ms/step
# [96~101]의 예측값 :  [[103.24158 ]
#  [105.244514]
#  [107.20731 ]]