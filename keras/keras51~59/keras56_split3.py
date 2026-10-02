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

# 목표 loss <= 0.1
# 결과는 [101, 102, 103, 104, 105, 106]의 근사치가 나오면 됨

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, 6)
# print(bbb)
# print(bbb.shape)    # (95, 6)

x = bbb[:-1, :-1]
y = bbb[:-1, -1]
# print(x, y)
# print(x.shape, y.shape) # (94, 5) (94,)
x = x.reshape(94,5,1)
# print(x.shape)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=100, input_shape=(5,1)))  # x.shape가 (94, 5, 1)인 데이터이므로. input_shape : (94, 5, 1) -> (5, 1)
model.add(Dense(100, activation='sigmoid'))
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
filepath = "".join([path, "k56_3_", date, "-", filename])

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
x_predict = x_predict.reshape(-1,5,1)
y_pred = model.predict(x_predict)

print("[96~101]의 예측값", y_pred)

# Results
# Epoch 371: early stopping
# 3/3 [==============================] - 0s 0s/step - loss: 7.8346e-04
# loss :  0.000783461204264313
# 1/1 [==============================] - 0s 95ms/step
# [96~101]의 예측값 [[100.89577]
#  [101.83861]
#  [102.72341]
#  [103.61347]
#  [104.44897]
#  [105.25564]]