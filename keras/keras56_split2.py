from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import datetime

#1. 데이터
a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]
              ]).T
# print(a.shape)  # (10, 2)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

arr = split_x(a, 3)
# print(arr)
# [[[ 1  9]
#   [ 2  8]
#   [ 3  7]]

#  [[ 2  8]
#   [ 3  7]
#   [ 4  6]]
# ...
#  [[ 8  2]
#   [ 9  1]
#   [10  0]]]
# print(arr.shape)    # (8, 3, 2)

# def split(dataset, size):
#     x = np.empty((len(dataset) - size + 1, size, dataset.shape[1]), dtype=int)
#     for i in range(0, len(dataset) - size + 1):
#         for j in range(0, size):
#             for k in range(0, dataset.shape[1]):
#                 print("i, j, k : ", i, j, k)
#                 print("dataset[i][k] : ", dataset[i][k])
#                 x[i][j][k] = dataset[i+j][k]

# arr2 = split(a, 3)
# print(arr2)
# print(arr2.shape)
# exit()

# x = arr[0:7]
# y = a.T[1][3:]
# print(x.shape, y.shape)  # (7,3,2) (7,)
# print(x, y)    # [6 5 4 3 2 1 0]

# 실습 코드. 5개씩 잘라서 위 4개를 x로 아래 5개 중 오른쪽 값을 y로
bbb = split_x(a, 5)
print(bbb)
print(bbb.shape)    # (6, 5, 2)

x = bbb[:, :-1]
# x = bbb[:, :-1, :] # 위는 여기서 끝 생략한 것

y = bbb[:, -1, 1]
# y = bbb[:, -1, -1]    # 1은 feature가 2라는 걸 알 때. 모르면 -1.

print(x.shape, y.shape)  # (6, 4, 2) (6,)
print(x, y)    # [6 5 4 3 2 1 0]
exit()
#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=200, input_shape=(3,2)))  # x.shape가 (8,3,2)인 데이터이므로. input_shape : (8,3,2) -> (3,2)
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
filepath = "".join([path, "k56_2_", date, "-", filename])

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

x_predict = np.array([[8,2],[9,1],[10,0]])
x_predict = x_predict.reshape(-1, 3, 2)
y_pred = model.predict(x_predict)

print("[[8,2],[9,1],[10,0]]의 예측값", y_pred)

# Results
# Epoch 84/3000
# 1/6 [====>.........................] - ETA: 0s - loss: 0.7583Restoring model weights from the end of the best epoch: 34.
# 6/6 [==============================] - 0s 5ms/step - loss: 0.3405 - val_loss: 0.7331 - lr: 0.0012
# Epoch 84: early stopping
# 1/1 [==============================] - 0s 92ms/step - loss: 0.3043
# loss :  0.30429795384407043
# 1/1 [==============================] - 0s 85ms/step
# [[8,2],[9,1],[10,0]]의 예측값 [[-0.25890496]]