from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, BaggingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np

#1. 데이터
x, y = load_diabetes(return_X_y=True)

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
xgb = XGBRegressor()
lg = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators=[('XGB', xgb), ('LGBM', lg), ('RF', rf), ('CAT', cat)],
    # voting="hard"   # voting 파라미터 없음
    # voting="soft"  
)

#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 :", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

# Results - VotingRegressor
# 999:    learn: 1.5862849        total: 947ms    remaining: 0us
# 최종 점수 : 0.39707811541894833
# r2 : 0.3970781154189483