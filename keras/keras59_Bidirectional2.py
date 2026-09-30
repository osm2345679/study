# 55-2 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Bidirectional
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

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델 구성
# model = Sequential()
# model.add(Bidirectional(SimpleRNN(units=200), input_shape=(3,1)))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1))

model = Sequential()
model.add(Bidirectional(SimpleRNN(units=200), input_shape=(3,1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)
path = './_save/keras55/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k55_2_", date, "-", filename])

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
model.fit(
    x, y,
    epochs=3000,
    batch_size=1,
    verbose=1,
    validation_split=0.01,
    callbacks=[es, mcp, rlr]
)

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)

x_predict = x_predict.reshape(-1, 3, 1)
y_pred = model.predict(x_predict)

print("[50,60,70]의 예측값", y_pred)

# Results
# Epoch 59/3000
#  1/12 [=>............................] - ETA: 0s - loss: 2.8600Restoring model weights from the end of the best epoch: 9.
# 12/12 [==============================] - 0s 3ms/step - loss: 20.6372 - val_loss: 118.6009 - lr: 0.0025
# Epoch 59: early stopping
# 1/1 [==============================] - 0s 86ms/step - loss: 12.2225
# loss :  12.222495079040527
# 1/1 [==============================] - 0s 85ms/step
# [50,60,70]의 예측값 [[78.24288]]

# Bidirectional
# Epoch 67/3000
#  1/12 [=>............................] - ETA: 0s - loss: 4.3861e-05Restoring model weights from the end of the best epoch: 17.
# 12/12 [==============================] - 0s 5ms/step - loss: 6.4441 - val_loss: 101.9977 - lr: 0.0025
# Epoch 67: early stopping
# 1/1 [==============================] - 0s 117ms/step - loss: 10.6005
# loss :  10.600461959838867
# 1/1 [==============================] - 0s 126ms/step
# [50,60,70]의 예측값 [[79.7456]]
# Epoch 99/3000
#  1/12 [=>............................] - ETA: 0s - loss: 2.0872e-05Restoring model weights from the end of the best epoch: 49.
# 12/12 [==============================] - 0s 4ms/step - loss: 1.2766e-04 - val_loss: 62.6201 - lr: 0.0012
# Epoch 99: early stopping
# 1/1 [==============================] - 0s 145ms/step - loss: 3.1282
# loss :  3.1282119750976562
# 1/1 [==============================] - 0s 130ms/step
# [50,60,70]의 예측값 [[65.14861]]