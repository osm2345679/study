# m06-1 카피
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import time

from sklearn.utils import all_estimators
import sklearn as sk

import warnings

warnings.filterwarnings("ignore")

#1. 데이터
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]
data_name = ['calfornia', 'diabetes']

#2. 모델 구성
all = all_estimators(type_filter="regressor")

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

start = time.perf_counter()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("===========", data_name[index], "============")
    model_name = []
    model_r2 = []

    for name, model in all:
        try:
            model = model()
            score = cross_val_score(model, x_train, y_train, cv=kfold)

            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            r2 = r2_score(y_test, y_predict)

            model_name.append(name)
            model_r2.append(r2)
        except:
            # print(name, "모델 오류")
            continue

    max_index = np.argmax(model_r2)
    # max_index = model_acc.index(max(model_acc))
    print("최고 모델 :", model_name[max_index], "R2 :", model_r2[max_index])
    
end = time.perf_counter()

print("걸린 시간 :", round(end-start, 2), "초")