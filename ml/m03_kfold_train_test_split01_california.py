from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import numpy as np

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

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
# r2 : [0.8255197  0.83371553 0.8329995  0.82694072 0.8320946 ] 
# 평균 r2 : 0.83
# y_predict r2 : 0.7721720817952313