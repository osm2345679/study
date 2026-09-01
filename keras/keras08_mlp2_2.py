import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(x)    # [0 1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 10)) # [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(x)    # [1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 11)) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(x)    # [1 2 3 4 5 6 7 8 9, 10]

x = np.array([range(10), range(21, 31,), range(201, 211)]).T
print(x)    # [[  0   1   2   3   4   5   6   7   8   9]
            #  [ 21  22  23  24  25  26  27  28  29  30]    -> [[  0  21 201] ... [  9  30 210]]
            #  [201 202 203 204 205 206 207 208 209 210]]
print(x.shape)  # (3, 10) -> (10, 3)

y = np.array(range(1,11))
print(y.shape)  # (10,)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=30)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

results = model.predict(np.array([[10, 31, 211]]))
print("[10, 31, 211]의 예측값 : ", results)

# Results

# Epoch 200/200
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 1.3225e-06
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 1.4337e-06
# loss :  1.4337263110064669e-06
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step
# [10, 31, 211]의 예측값 :  [[10.997751]]