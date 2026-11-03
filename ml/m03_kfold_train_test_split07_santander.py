from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score,cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

#1. 데이터
path = 'c:/study/_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop('target', axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=42,
    shuffle=True,
    stratify=y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델 구성
# model1 = DecisionTreeClassifier(random_state=random_state)
# model2 = RandomForestClassifier(random_state=random_state)
# model3 = GradientBoostingClassifier(random_state=random_state)
model = XGBClassifier()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("acc :", scores, "\n평균 acc :", round(np.mean(scores), 2))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_predict)
print("y_predict acc :", acc)
# acc : [0.9139     0.9106     0.9129     0.91226667 0.91106667] 
# 평균 acc : 0.91
# y_predict acc : 0.90716