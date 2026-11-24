# Pseudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾음
# 스태킹 : 모델을 돌려 나온 것으로 구성해서 새로운 데이터셋을 만듦
#       : 한 데이터로 여러 모델을 돌려서 돌리는 것으로 칼럼을 만듬.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, BaggingClassifier, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.8,
    random_state=42,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
xgb = XGBRegressor()
lg = LGBMRegressor(verbosity=-1)
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

models = [xgb, lg, rf, cat]
train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print(f"{class_name} R2 : {score:.4f}")

x_test_new = np.array(test_list).T
x_train_new = np.array(train_list).T

model2 = CatBoostRegressor(verbose=0)

#3. 컴파일, 훈련
model2.fit(x_train_new, y_train)

#4. 평가, 예측
y_pred_new = model2.predict(x_test_new)
r2 = r2_score(y_test, y_pred_new)
print("Total Result :", r2)

# Results
# XGBRegressor R2 : 0.7894
# LGBMRegressor R2 : 0.8121
# RandomForestRegressor R2 : 0.7696
# CatBoostRegressor R2 : 0.8249
# Total Result : 0.7694577461889819