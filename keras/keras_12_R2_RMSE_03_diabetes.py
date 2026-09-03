from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
#R2 > 0.62

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42
)

print(x_train.shape, x_test.shape, y_train.shape, y_train.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

#4. 평가, 예측
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
print("rmse: ", rmse)

# Result

# Epoch 500/500
# 10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2952.8862 
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2825.1602 
# loss :  2825.16015625
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
# r2 :  0.4766581996939855
# mse :  2825.160102711309