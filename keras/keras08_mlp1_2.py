import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5],    # (2,5)
              [6,7,8,9,10]])
x = x.T  # 행렬 변환 - T 또는 transpose() 사용. numpy 라이브러리 함수.
y = np.array([1,2,3,4,5])   # (5,)

print(x.shape)  # (5, 2)
print(y.shape)  # (5,)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))    # 행 무시 열 우선
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

results = model.predict(np.array([[6,11]])) # (n, 2) 형태로 줘야 함. np.array[6,11] 하면 오류남.
print("[6, 11]의 예측값 : ", results)

# Results

# Epoch 100/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0386
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 0.0375
# loss :  0.03745635598897934
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
# [6, 11]의 예측값 :  [[5.614131]]