import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]).transpose()
print(x.shape, y.shape) # (10, 1) (10, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=10)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

results = model.predict(np.array([5]))
print("[10]의 예측값 : ", results)

# Results

# Epoch 1000/1000
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 6.1951e-05
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 6.0856e-05
# loss :  6.085566565161571e-05
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
# [10]의 예측값 :  [[ 1.0988978e+01 -6.5974891e-03 -9.9110341e-01]]