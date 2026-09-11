# 11-3 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss(mse) : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)    # 평가, 예측 안 함. 그냥 실제값(리스트), 예측값(리스트) 넣으면 r2 score 계산해주는 함수. '1-SSE/SST'. 'MSE/Var(y)'
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mss : ", mse)

def RMSE(y_test, y_predict) :   # RMSE함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# Results
#
# Epoch 1000/1000
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 26.1312 
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 25.0293 
# loss(mse) :  25.02928924560547
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
# r2 :  0.6993256621610193
# mss :  25.029289032859303
# RMSE :  5.002928045940627
