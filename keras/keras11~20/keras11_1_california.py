# import ssl
# ssl._create_defalut_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    random_state=42
)

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=8))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=16)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# Results
#
# Epoch 200/200
# 903/903 ━━━━━━━━━━━━━━━━━━━━ 1s 577us/step - loss: 0.6296
# 194/194 ━━━━━━━━━━━━━━━━━━━━ 0s 591us/step - loss: 0.6011
# loss :  0.6010539531707764