# Pseudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾음
# 스태킹 : 모델을 돌려 나온 것으로 구성해서 새로운 데이터셋을 만듦
#       : 한 데이터로 여러 모델을 돌려서 돌리는 것으로 칼럼을 만듬.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, BaggingClassifier, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.8,
    random_state=42,
    # stratify=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
xgb = XGBClassifier()
lg = LGBMClassifier(verbosity=-1)
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

models = [xgb, lg, rf, cat]
train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print(f"{class_name} ACC : {score:.4f}")

x_test_new = np.array(test_list).T
# print(x_test_new.shape) # (456, 4)

x_train_new = np.array(train_list).T
# print(x_train_new.shape)    # (113, 4)

model2 = CatBoostClassifier(verbose=0)
# model2 = LogisticRegression()

#3. 컴파일, 훈련
model2.fit(x_train_new, y_train)

#4. 평가, 예측
y_pred_new = model2.predict(x_test_new)
acc = accuracy_score(y_test, y_pred_new)
print("Total Result :", acc)

# Results
# XGBClassifier ACC : 0.9496
# LGBMClassifier ACC : 0.9539
# RandomForestClassifier ACC : 0.9518
# CatBoostClassifier ACC : 0.9671
# Total Result : 0.9649122807017544