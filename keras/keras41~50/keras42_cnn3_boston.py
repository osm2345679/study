# 20-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import datetime

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)

#2. 모델 구성
model = Sequential()
# model.add(Dense(32, input_dim=13))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dropout(0.2))
# model.add(Dense(8))
# model.add(Dropout(0.2))
# model.add(Dense(1))

model.add(Conv2D(64, (2,1), input_shape=(13,1,1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
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
    patience = 150,
    restore_best_weights=True
)

hist = model.fit(x_train, y_train, epochs=2000, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6, 4))
plt.title('보스턴 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# Results
# Epoch 750/2000
# 31/31 [==============================] - 0s 3ms/step - loss: 47.5473 - val_loss: 76.1807
# 4/4 [==============================] - 0s 24ms/step - loss: 55.0602
# loss :  55.06015396118164