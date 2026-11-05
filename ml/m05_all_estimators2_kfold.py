# m04-1 카피
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
import numpy as np

from sklearn.utils import all_estimators
import sklearn as sk

import warnings

warnings.filterwarnings("ignore")

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델 구성
all = all_estimators(type_filter="regressor")

for name, model in all:
    try:
        model = model()
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, "의 train r2 :", score)

        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        r2 = r2_score(y_test, y_predict)
        print(name, "의 r2 :", r2)
    except:
        print(name, "모델 오류")

# Results
# ARDRegression 의 train r2 : [0.59688042 0.59699829 0.62143443 0.59068566 0.59633404]
# ARDRegression 의 r2 : 0.5560245593958151
# AdaBoostRegressor 의 train r2 : [0.4626617  0.38635217 0.40716217 0.46244702 0.52224905]
# AdaBoostRegressor 의 r2 : 0.467565354030692
# BaggingRegressor 의 train r2 : [0.78218088 0.78906714 0.78500722 0.77717109 0.79194196]
# BaggingRegressor 의 r2 : 0.7126141793413523
# BayesianRidge 의 train r2 : [0.60969226 0.6042173  0.63543146 0.60073314 0.60723159]
# BayesianRidge 의 r2 : 0.5618386907322941
# CCA 모델 오류
# DecisionTreeRegressor 의 train r2 : [0.58377852 0.6309784  0.60131525 0.57665934 0.56980329]
# DecisionTreeRegressor 의 r2 : 0.49183543396712703
# DummyRegressor 의 train r2 : [-4.58127659e-04 -6.38358401e-05 -4.37492967e-07 -4.91472624e-04
#  -6.29951876e-05]
# DummyRegressor 의 r2 : -0.0003124538448660008
# ElasticNet 의 train r2 : [0.42708313 0.42170746 0.43999508 0.41495041 0.43015619]
# ElasticNet 의 r2 : 0.4060779070358749
# ElasticNetCV 의 train r2 : [0.58124522 0.57556236 0.61620489 0.57567083 0.58365458]
# ElasticNetCV 의 r2 : 0.5694270553195374
# ExtraTreeRegressor 의 train r2 : [0.60257007 0.56240426 0.5906791  0.60727816 0.60989495]
# ExtraTreeRegressor 의 r2 : 0.40704202888004515
# ExtraTreesRegressor 의 train r2 : [0.80619587 0.80519879 0.81947914 0.79719436 0.8109041 ]
# ExtraTreesRegressor 의 r2 : 0.7525985852142747
# GammaRegressor 의 train r2 : [-4.70629350e-04 -6.36025757e-05 -4.40741796e-07 -4.85059096e-04
#  -6.34683959e-05]
# GammaRegressor 의 r2 : -0.0003124538448660008