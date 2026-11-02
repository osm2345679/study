from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델 구성
model = DecisionTreeClassifier()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)

print("acc :", scores, "\n평균 acc :", round(np.mean(scores), 2))
# acc : [1.         0.96666667 0.93333333 0.93333333 0.93333333] 
# 평균 acc : 0.95