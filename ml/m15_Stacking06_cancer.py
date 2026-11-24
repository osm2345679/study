# Pseudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾음
# 스태킹 : 모델을 돌려 나온 것으로 구성해서 새로운 데이터셋을 만듦
#       : 한 데이터로 여러 모델을 돌려서 돌리는 것으로 칼럼을 만듬.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, StackingRegressor
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

model = StackingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('RF', rf), ('CAT', cat)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5
)

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Total Result :", acc)

# Results
# XGBClassifier ACC : 0.9496
# LGBMClassifier ACC : 0.9539
# RandomForestClassifier ACC : 0.9518
# CatBoostClassifier ACC : 0.9671
# Total Result : 0.9649122807017544