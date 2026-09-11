import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200)

#4. 평가, 예측
result = model.predict(np.array([4]))
print("4의 예측값 : ", result)

# Result
#
# Epoch 200/200
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0037
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
# 4의 예측값 :  [[4.159524]]