from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, BaggingClassifier, RandomForestClassifier
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
# model = DecisionTreeClassifier()
# model = BaggingClassifier(   # Bootstrap + Aggregating. sample 비율만큼 뽑아서 여러 번 해서 평균(회귀) 또는 다수결(분류)로 결과 냄. 
#     DecisionTreeClassifier(),
#     n_estimators=10,
#     random_state=42,
#     n_jobs=-1,
#     max_samples=0.8,
#     # bootstrap=True, # 디폴트. 중복 허용.
#     bootstrap=False   # 중복 허용 X -> Pasting 기법
# )
model = RandomForestClassifier()

#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종 점수 :", result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc :", acc)

# Results - DecisionTreeClassifier
# 최종 점수 : 0.8881578947368421
# acc : 0.8881578947368421

# Results - BaggingClassifier. bootstrap = True
# 최종 점수 : 0.9320175438596491
# acc : 0.9320175438596491

# Results - BaggingRegressor. bootstrap = false -> Pasting
# 최종 점수 : 0.9254385964912281
# acc : 0.9254385964912281

# Results - RandomForestRegressor
# 최종 점수 : 0.9605263157894737
# acc : 0.9605263157894737