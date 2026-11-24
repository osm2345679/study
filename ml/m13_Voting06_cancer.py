from sklearn.datasets import load_breast_cancer
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
lg = LGBMClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('XGB', xgb), ('LGBM', lg), ('RF', rf), ('CAT', cat)],
    # voting="hard"   # 디폴트. 각 모델 결과로 다수결(가중치 동일한 경우)
    voting="soft"   # 각 클래스별 확률 평균 구해서 확률 높은 결로 최종 결과 냄.  
)

#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 :", result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc :", acc)

# Results - VotingClassifier - hard voting
# 999:    learn: 0.0289477        total: 1.98s    remaining: 0us
# 최종 점수 : 0.9649122807017544
# acc : 0.9649122807017544

# Results - VotingClassifier - soft voting
# 999:    learn: 0.0289477        total: 2.12s    remaining: 0us
# 최종 점수 : 0.9692982456140351
# acc : 0.9692982456140351