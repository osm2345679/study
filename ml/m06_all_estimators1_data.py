# m03-00 카피
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import time

from sklearn.utils import all_estimators
import sklearn as sk

import warnings

warnings.filterwarnings("ignore")

#1. 데이터
iris = load_iris(return_X_y=True)
# print(iris)

cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['iris', 'cancer', 'wine', 'digits']

#2. 모델 구성
all = all_estimators(type_filter="classifier")

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

start = time.perf_counter()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("===========", data_name[index], "============")
    model_name = []
    model_acc = []

    for name, model in all:
        try:
            model = model()
            score = cross_val_score(model, x_train, y_train, cv=kfold)

            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_predict)

            model_name.append(name)
            model_acc.append(acc)
        except:
            # print(name, "모델 오류")
            continue

    max_index = np.argmax(model_acc)
    # max_index = model_acc.index(max(model_acc))
    print("최고 모델 :", model_name[max_index], "ACC :", model_acc[max_index])
    
end = time.perf_counter()

print("걸린 시간 :", round(end-start, 2), "초")

# Results
# =========== iris ============
# 최고 모델 : LinearDiscriminantAnalysis ACC : 0.9666666666666667
# =========== cancer ============
# 최고 모델 : AdaBoostClassifier ACC : 0.956140350877193
# =========== wine ============
# 최고 모델 : GaussianNB ACC : 0.9722222222222222
# =========== digits ============
# 최고 모델 : ExtraTreesClassifier ACC : 0.9472222222222222
# 걸린 시간 : 156.11 초