from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import datetime

#1. 데이터
path="./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델 구성
model = XGBRegressor()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("r2 :", scores, "\n평균 r2 :", round(np.mean(scores), 2))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)
print("y_predict r2 :", r2)
# r2 : [0.29115063 0.30859363 0.26556945 0.31608206 0.3223601 ] 
# 평균 r2 : 0.3
# y_predict r2 : 0.1792140007019043