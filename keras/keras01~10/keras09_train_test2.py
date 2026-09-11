import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

x_train = x[0:7]    # [start:end:step]
y_train = y[:7] # start 디폴트는 0

x_test = x[7:10]
y_test = x[7:]  # end 디폴트는 리스트 끝

print(x_train, y_train, x_test, y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# Results
# 
# Epoch 500/500
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 1.9949e-04 
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - loss: 7.4128e-04
# loss :  0.0007412821869365871