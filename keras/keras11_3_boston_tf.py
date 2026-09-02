from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=10)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# Results
#
# Epoch 1000/1000
# 41/41 ━━━━━━━━━━━━━━━━━━━━ 0s 898us/step - loss: 28.4356
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 22.9518 
# loss :  22.95181655883789
