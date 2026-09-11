# 9-1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5,
          verbose = -1
          )
# verbose = 0 : 침묵
# verbose = 1 : 디폴트
# verbose = 2 : 프로그레스바 삭제
# verbose = 나머지 : epochs만


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# Results
# 
# Epoch 498/500
# Epoch 499/500
# Epoch 500/500
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 5.0289e-05
# loss :  5.028881787438877e-0