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
    test_size=0.2,
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
model.fit(x_train, y_train, epochs=2400, batch_size=256, verbose=1, validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# Results

# Epoch 2400/2400
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step - loss: 2779.1367 - val_loss: 3276.5144
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 2938.9602 
# loss :  2938.960205078125
# PS C:\study> 