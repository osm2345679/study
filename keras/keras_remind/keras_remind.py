from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터

# tensorflow의 데이터셋을 활용하는 경우
# from tensorflow.keras.datasets import boston_housing

# (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    random_state=42
)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=8))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

results = model.predict(x_test)
print("x_test의 예측값 : ", results)

plt.plot(results, color='red')
plt.show()