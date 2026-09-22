# 28-2 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.25,
    random_state=42
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,5,2,1)
x_test = x_test.reshape(-1,5,2,1)
x_val = x_val.reshape(-1,5,2,1)

#2. 모델 구성
model = Sequential()
# model.add(Dense(32, input_dim=10))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dropout(0.2))
# model.add(Dense(4))
# model.add(Dropout(0.2))
# model.add(Dense(2))
# model.add(Dropout(0.2))
# model.add(Dense(1))

model.add(Conv2D(64, (2,1), input_shape=(5,2,1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,1), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())    # 
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(1))  # (10,)

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = min,
    patience = 70,
    restore_best_weights = False
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=2400, batch_size=256, verbose=1, validation_data=(x_val, y_val), callbacks=[es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,4))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.legend(loc='upper right')
plt.title('Diabetes Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.grid()
plt.show()

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 275/2400
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - loss: 5307.4209 - val_loss: 4256.1855
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 4309.0347
# loss :  4309.03466796875
# 걸린 시간 :  35.0 초