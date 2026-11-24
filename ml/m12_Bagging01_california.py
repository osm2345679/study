from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import numpy as np

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

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
# model = DecisionTreeRegressor()
# model = BaggingRegressor(   # Bootstrap + Aggregating. sample 비율만큼 뽑아서 여러 번 해서 평균(회귀) 또는 다수결(분류)로 결과 냄. 
#     DecisionTreeRegressor(),
#     n_estimators=10,
#     random_state=42,
#     n_jobs=-1,
#     max_samples=0.8,
#     # bootstrap=True, # 디폴트. 중복 허용.
#     bootstrap=False   # 중복 허용 X
# )
model = RandomForestRegressor()

#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 :", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

# Results - DecisionTreeRegressor
# 최종 점수 : 0.5544234773191575
# r2 : 0.5544234773191575

# Results - BaggingRegressor. bootstrap = True
# 최종 점수 : 0.7380731549277775
# r2 : 0.7380731549277775

# Results - BaggingRegressor. bootstrap = false
# 최종 점수 : 0.7440975467284875
# r2 : 0.7440975467284875

# Results - RandomForestRegressor
# 최종 점수 : 0.7695475436838715
# r2 : 0.7695475436838715