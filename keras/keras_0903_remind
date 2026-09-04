from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#1. 데이터
path = "./_data/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col=0)

print(train_csv)
print(train_csv.shape)

# 이상치, 결측치 처리
train_csv = train_csv.dropna()

# train, test 분리. test_csv은 validation용
x = train_csv.drop(['count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print("==============================")
print(x)
print("==============================")
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    random_state=42
)

#2. 모델 구성
model = Sequential()
model.add(Dense(18, input_dim=10))
model.add(Dense(27))
model.add(Dense(36))
model.add(Dense(18))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)
