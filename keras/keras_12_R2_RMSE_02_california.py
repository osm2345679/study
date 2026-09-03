from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# r2 > 0.55

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

def RMSE(y_test, y_predict) :
    mse = mean_squared_error(y_test, y_predict)
    return np.sqrt(mse)

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)

# Result

# Epoch 1000/1000
# 452/452 ━━━━━━━━━━━━━━━━━━━━ 0s 656us/step - loss: 0.5989
# 194/194 ━━━━━━━━━━━━━━━━━━━━ 0s 592us/step - loss: 0.5955
# loss :  0.5954731106758118
# 194/194 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step
# r2 :  0.5463200350842436
# mse :  0.5954731500635166
# rmse :  0.771669067712006