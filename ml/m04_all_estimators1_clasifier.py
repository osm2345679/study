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
all = all_estimators(type_filter="classifier")
# print(all)
print("sk version :", sk.__version__)   # sk version : 1.9.1
print("모델의 개수 :", len(all))    # 모델의 개수 : 44

# all = all_estimators(type_filter="regressor")
# print(all)
print("모델의 개수 :", len(all))    # 모델의 개수 : 55

for name, model in all:
    try:
        model = model()

        #3. 훈련
        model.fit(x_train, y_train)

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, "의 acc :", acc)
    except:
        print(name, " 모델 오류")

# Results
# sk version : 1.9.1
# 모델의 개수 : 44
# 모델의 개수 : 44
# AdaBoostClassifier 의 acc : 0.9333333333333333
# BaggingClassifier 의 acc : 0.9666666666666667
# BernoulliNB 의 acc : 0.7333333333333333
# CalibratedClassifierCV 의 acc : 0.8333333333333334
# CategoricalNB  모델 오류
# ClassifierChain  모델 오류
# ComplementNB  모델 오류
# DecisionTreeClassifier 의 acc : 0.9333333333333333
# DummyClassifier 의 acc : 0.3333333333333333
# ExtraTreeClassifier 의 acc : 0.9
# ExtraTreesClassifier 의 acc : 0.9666666666666667
# FixedThresholdClassifier  모델 오류
# GaussianNB 의 acc : 0.9666666666666667
# GaussianProcessClassifier 의 acc : 0.9333333333333333
# GradientBoostingClassifier 의 acc : 0.9666666666666667
# HistGradientBoostingClassifier 의 acc : 0.9
# KNeighborsClassifier 의 acc : 0.9333333333333333
# LabelPropagation 의 acc : 0.9666666666666667
# LabelSpreading 의 acc : 0.9666666666666667
# LinearDiscriminantAnalysis 의 acc : 1.0
# LinearSVC 의 acc : 0.9
# LogisticRegression 의 acc : 0.9333333333333333
# LogisticRegressionCV 의 acc : 0.9333333333333333
# MLPClassifier 의 acc : 0.9333333333333333
# MultiOutputClassifier  모델 오류
# MultinomialNB  모델 오류
# NearestCentroid 의 acc : 0.8333333333333334
# NuSVC 의 acc : 0.9666666666666667
# OneVsOneClassifier  모델 오류
# OneVsRestClassifier  모델 오류
# OutputCodeClassifier  모델 오류
# PassiveAggressiveClassifier 의 acc : 0.9333333333333333
# Perceptron 의 acc : 0.9
# QuadraticDiscriminantAnalysis 의 acc : 1.0
# RadiusNeighborsClassifier 의 acc : 0.9
# RandomForestClassifier 의 acc : 0.9666666666666667
# RidgeClassifier 의 acc : 0.7666666666666667
# RidgeClassifierCV 의 acc : 0.7666666666666667
# SGDClassifier 의 acc : 0.9333333333333333
# SVC 의 acc : 0.9666666666666667
# SelfTrainingClassifier  모델 오류
# StackingClassifier  모델 오류
# TunedThresholdClassifierCV  모델 오류
# VotingClassifier  모델 오류