# m03-00 카피
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.utils import all_estimators
import sklearn as sk

import warnings

warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_iris(return_X_y=True)

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

#2. 모델 구성
all = all_estimators(type_filter="regressor")

print("sk version :", sk.__version__)   # sk version : 1.9.1
print("모델의 개수 :", len(all))    # 모델의 개수 : 44

for name, model in all:
    try:
        model = model()

        #3. 컴파일, 훈련
        model.fit(x_train, y_train)

        #4. 평가, 예측
        r2 = model.score(x_test, y_test)
        print(name, "의 r2 :", r2)
    except:
        print(name, "모델 오류")
        
# Results
# sk version : 1.9.1
# 모델의 개수 : 55
# ARDRegression 의 r2 : 0.9134142860872644
# AdaBoostRegressor 의 r2 : 0.9446455437143053
# BaggingRegressor 의 r2 : 0.937
# BayesianRidge 의 r2 : 0.9138242358133756
# CCA 모델 오류
# DecisionTreeRegressor 의 r2 : 0.9
# DummyRegressor 의 r2 : 0.0
# ElasticNet 의 r2 : 0.44846656123994366
# ElasticNetCV 의 r2 : 0.9106008097343752
# ExtraTreeRegressor 의 r2 : 0.95
# ExtraTreesRegressor 의 r2 : 0.956785
# GammaRegressor 모델 오류
# GaussianProcessRegressor 의 r2 : -1.5
# GradientBoostingRegressor 의 r2 : 0.9327012973157442
# HistGradientBoostingRegressor 의 r2 : 0.9108756888796392
# HuberRegressor 의 r2 : 0.9123792633359767
# IsotonicRegression 모델 오류
# KNeighborsRegressor 의 r2 : 0.938
# KernelRidge 의 r2 : -0.6862487382168276
# Lars 의 r2 : 0.9139814715607812
# LarsCV 의 r2 : 0.9102910139968801
# Lasso 의 r2 : 0.0
# LassoCV 의 r2 : 0.9102936220111926
# LassoLars 의 r2 : 0.0
# LassoLarsCV 의 r2 : 0.9102910139968801
# LassoLarsIC 의 r2 : 0.9139814715607812
# LinearRegression 의 r2 : 0.9139814715607814
# LinearSVR 의 r2 : 0.9117172558364444
# MLPRegressor 의 r2 : 0.9305880247449041
# MultiOutputRegressor 모델 오류
# MultiTaskElasticNet 모델 오류
# MultiTaskElasticNetCV 모델 오류
# MultiTaskLasso 모델 오류
# MultiTaskLassoCV 모델 오류
# NuSVR 의 r2 : 0.9491046034315241
# OrthogonalMatchingPursuit 의 r2 : 0.8920132525993715
# OrthogonalMatchingPursuitCV 의 r2 : 0.9139814715607812
# PLSCanonical 모델 오류
# PLSRegression 의 r2 : 0.9136567363116532
# PassiveAggressiveRegressor 의 r2 : 0.869645808336948
# PoissonRegressor 의 r2 : 0.6827581975491651
# QuantileRegressor 의 r2 : 0.0
# RANSACRegressor 의 r2 : 0.913577774111747
# RadiusNeighborsRegressor 의 r2 : 0.9117846927099141
# RandomForestRegressor 의 r2 : 0.951955
# RegressorChain 모델 오류
# Ridge 의 r2 : 0.9136220271212744
# RidgeCV 의 r2 : 0.9136220271212744
# SGDRegressor 의 r2 : 0.8997956470274804
# SVR 의 r2 : 0.9427371575310685
# StackingRegressor 모델 오류
# TheilSenRegressor 의 r2 : 0.9061385687755568
# TransformedTargetRegressor 의 r2 : 0.9139814715607814
# TweedieRegressor 의 r2 : 0.8278124183920293
# VotingRegressor 모델 오류