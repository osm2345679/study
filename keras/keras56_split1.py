from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import datetime

#1. 데이터

a = np.array(range(1,11))
size = 5    # timestep 사이즈

print(a.shape)  # (10,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape)   # (6,5)


'''
직접 짜본 코드
split2에서 보니 배열 사이즈 2차원이라 3차원 배열 넣으면 오류남.
배열 말고 리스트로 해야 할듯

# print(x.shape)
# exit()

# print(a[0:3])
# x[0] = a[0:5]
# print(x)

x = np.empty((np.size(a) - size + 1, size), dtype=int)  # np.size()는 전체 갯수 반환하므로 np.size(a[0])으로 해야 첫 행의 갯수만 반환하는 len()과 동일

print(type(x))  # <class 'numpy.ndarray'>. aaa = []는 list, x는 ndarray임.

for i in range(0, np.size(a) - size + 1):
    x[i] = a[i:i+size]
    
print(x)

'''

x = bbb.T[0:4].T
y = bbb.T[4]
print(x, y)
print(x.shape, y.shape) # (6, 4) (6,)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=200, input_shape=(4,1)))  # x.shape가 (6,4,1)인 데이터이므로. input_shape : (6,4,1) -> (4,1)
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)
path = './_save/keras56/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k56_1_", date, "-", filename])

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
    patience=20,
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

x_predict = np.array([7,8,9,10])
x_predict = x_predict.reshape(-1, x_predict.size, 1)
y_pred = model.predict(x_predict)

print("[7,8,9,10]의 예측값", y_pred)

# Results
# Epoch 60/3000
# 1/5 [=====>........................] - ETA: 0s - loss: 2.8361Restoring model weights from the end of the best epoch: 10.
# 5/5 [==============================] - 0s 7ms/step - loss: 3.9224 - val_loss: 7.1478 - lr: 0.0025
# Epoch 60: early stopping
# 1/1 [==============================] - 0s 86ms/step - loss: 8.9750
# loss :  8.975030899047852
# 1/1 [==============================] - 0s 85ms/step
# [7,8,9,10]의 예측값 [[10.055873]]