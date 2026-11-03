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
# AdaBoostClassifier 의 train acc : [0.95833333 0.83333333 0.91666667 0.95833333 0.91666667]
# AdaBoostClassifier 의 acc : 0.9333333333333333
# BaggingClassifier 의 train acc : [0.91666667 0.875      0.91666667 0.95833333 0.91666667]
# BaggingClassifier 의 acc : 0.9
# BernoulliNB 의 train acc : [0.70833333 0.75       0.875      0.70833333 0.79166667]
# BernoulliNB 의 acc : 0.7333333333333333
# CalibratedClassifierCV 의 train acc : [0.875      0.95833333 0.70833333 0.875      1.        ]
# CalibratedClassifierCV 의 acc : 0.7333333333333333
# CategoricalNB 모델 오류
# ClassifierChain 모델 오류
# ComplementNB 모델 오류
# DecisionTreeClassifier 의 train acc : [0.91666667 0.875      0.91666667 0.95833333 0.875     ]
# DecisionTreeClassifier 의 acc : 0.9
# DummyClassifier 의 train acc : [0.29166667 0.29166667 0.20833333 0.33333333 0.20833333]
# DummyClassifier 의 acc : 0.1
# ExtraTreeClassifier 의 train acc : [0.91666667 0.95833333 0.91666667 1.         0.83333333]
# ExtraTreeClassifier 의 acc : 0.8333333333333334
# ExtraTreesClassifier 의 train acc : [0.91666667 0.91666667 0.91666667 1.         0.91666667]
# ExtraTreesClassifier 의 acc : 0.9333333333333333
# FixedThresholdClassifier 모델 오류
# GaussianNB 의 train acc : [0.95833333 0.95833333 0.91666667 1.         0.95833333]
# GaussianNB 의 acc : 0.8666666666666667
# GaussianProcessClassifier 의 train acc : [0.95833333 1.         0.91666667 1.         0.91666667]
# GaussianProcessClassifier 의 acc : 0.8333333333333334
# GradientBoostingClassifier 의 train acc : [0.91666667 0.875      0.91666667 0.95833333 0.95833333]
# GradientBoostingClassifier 의 acc : 0.9
# HistGradientBoostingClassifier 의 train acc : [0.95833333 0.83333333 0.91666667 0.95833333 0.875     ]
# HistGradientBoostingClassifier 의 acc : 0.1
# KNeighborsClassifier 의 train acc : [0.91666667 0.91666667 0.95833333 1.         0.91666667]
# KNeighborsClassifier 의 acc : 0.9
# LabelPropagation 의 train acc : [0.875      1.         0.95833333 1.         0.91666667]
# LabelPropagation 의 acc : 0.8
# LabelSpreading 의 train acc : [0.875      1.         0.95833333 1.         0.91666667]
# LabelSpreading 의 acc : 0.8333333333333334
# LinearDiscriminantAnalysis 의 train acc : [0.95833333 1.         0.91666667 1.         1.        ]
# LinearDiscriminantAnalysis 의 acc : 0.9666666666666667
# LinearSVC 의 train acc : [0.95833333 1.         0.875      0.91666667 1.        ]
# LinearSVC 의 acc : 0.8
# LogisticRegression 의 train acc : [0.95833333 1.         0.91666667 1.         0.95833333]
# LogisticRegression 의 acc : 0.8666666666666667
# LogisticRegressionCV 의 train acc : [0.95833333 1.         0.91666667 1.         0.875     ]
# LogisticRegressionCV 의 acc : 0.9666666666666667
# MLPClassifier 의 train acc : [0.91666667 1.         0.91666667 0.95833333 0.91666667]
# MLPClassifier 의 acc : 0.9333333333333333
# MultiOutputClassifier 모델 오류
# MultinomialNB 모델 오류
# NearestCentroid 의 train acc : [0.79166667 0.79166667 0.91666667 0.83333333 0.91666667]
# NearestCentroid 의 acc : 0.8666666666666667
# NuSVC 의 train acc : [0.95833333 1.         0.91666667 1.         0.91666667]
# NuSVC 의 acc : 0.9
# OneVsOneClassifier 모델 오류
# OneVsRestClassifier 모델 오류
# OutputCodeClassifier 모델 오류
# PassiveAggressiveClassifier 의 train acc : [0.83333333 0.70833333 0.75       0.95833333 0.79166667]
# PassiveAggressiveClassifier 의 acc : 0.7666666666666667
# Perceptron 의 train acc : [0.79166667 0.91666667 0.75       0.91666667 0.95833333]
# Perceptron 의 acc : 0.7666666666666667
# QuadraticDiscriminantAnalysis 의 train acc : [0.95833333 1.         0.91666667 1.         0.91666667]
# QuadraticDiscriminantAnalysis 모델 오류
# RadiusNeighborsClassifier 의 train acc : [0.91666667        nan 0.91666667        nan 0.91666667]
# RadiusNeighborsClassifier 모델 오류
# RandomForestClassifier 의 train acc : [0.91666667 0.875      0.91666667 1.         0.95833333]
# RandomForestClassifier 의 acc : 0.9
# RidgeClassifier 의 train acc : [0.83333333 0.95833333 0.66666667 0.83333333 0.875     ]
# RidgeClassifier 의 acc : 0.7333333333333333
# RidgeClassifierCV 의 train acc : [0.83333333 0.95833333 0.66666667 0.83333333 0.875     ]
# RidgeClassifierCV 의 acc : 0.7
# SGDClassifier 의 train acc : [0.91666667 1.         0.875      0.91666667 0.875     ]
# SGDClassifier 의 acc : 0.6666666666666666
# SVC 의 train acc : [0.91666667 1.         0.91666667 1.         0.91666667]
# SVC 의 acc : 0.9
# SelfTrainingClassifier 모델 오류
# StackingClassifier 모델 오류
# TunedThresholdClassifierCV 모델 오류
# VotingClassifier 모델 오류