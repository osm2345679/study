from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import numpy as np

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델 구성
model = XGBRegressor()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)

print("r2 :", scores, "\n평균 r2 :", round(np.mean(scores), 2))
# r2 : [0.83116009 0.83522882 0.82451226 0.8490521  0.83352332] 
# 평균 r2 : 0.83