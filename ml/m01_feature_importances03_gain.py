#01 카피
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=random_state,
    stratify=y
)

#2. 모델 구성
# model1 = DecisionTreeClassifier(random_state=random_state)
# model2 = RandomForestClassifier(random_state=random_state)
# model3 = GradientBoostingClassifier(random_state=random_state)
model = XGBClassifier(random_state=random_state)

#3. 컴파일, 훈련
model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")    # ========= XGBClassifier ========

#4. 평가, 예측
score = model.score(x_test, y_test)    # 분류에선 accuracy_score, 회귀에선 r2_score
print("acc : ", score)  # acc :  0.9333333333333333

print("F 스코어 :", model.feature_importances_) # F스코어 : 빈도수
# F 스코어 : [0.02430454 0.02472077 0.7376847  0.21328993]

gain_scores = model.get_booster().get_score(importance_type='gain')
print("gain 스코어 :", gain_scores) # gain 스코어 : 손실함수의 개선 점수, 높으면 개선 많이 됐다는 뜻
# gain 스코어 : {'f0': 0.09280245006084442, 'f1': 0.09439174085855484, 'f2': 2.8167145252227783, 'f3': 0.8144086599349976}