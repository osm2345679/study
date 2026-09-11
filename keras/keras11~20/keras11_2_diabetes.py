from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42
)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2400, batch_size=256)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# Results

# Epoch 400/400
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 2970.0308 
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2788.3452 
# loss :  2788.34521484375