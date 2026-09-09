from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=8))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, )

#4. 평가, 예측