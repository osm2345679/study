from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1, input_dim=3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=3) # 훈련을 시킬 때 배치사이즈를 지정함. 데이터가 클 때는 필수. 작으면 더 가중치가 더 많이 업데이트 됨. 디폴트는 32.

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
# result = model.predict(np.array([1,2,3,4,5,6]))
# print("6의 예측값 : ", result)