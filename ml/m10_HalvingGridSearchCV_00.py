# 8-00 카피

import time
import warnings

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

parameters = [
    {"n_estimators" : [100, 500], "max_depth" : [6, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},  # 18번
    {"max_depth" : [6, 8, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},  # 12번
    {'min_chid_weight' : [2, 3, 5, 10], "learning_rate" : [0.1, 0.01, 0.001]} # 12번
]   # 42번

#2. 모델 구성
xgb = XGBClassifier()
model = HalvingGridSearchCV(
    xgb,
    parameters,
    cv=kfold,
    verbose=1,
    # n_jobs=-1,
    refit=True,
    min_resources=100, # 2 * 5 * 10 : 2 * cv * class갯수
    max_resources=1437, # train dataset의 행의 갯수
    factor=2
)

#3. 컴파일, 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)

#4. 평가, 예측
print("==========================================")
print("model.score :", model.score(x_test, y_test))
print("best_score :", model.best_score_)    # train 데이터로만 나온 결과. 과적합 되어 있을 확률 높음.

print("걸린 시간 :", round(end-start, 2), "초")

# Results
# n_iterations: 4
# n_required_iterations: 6
# n_possible_iterations: 4
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 21
# n_resources: 200
# Fitting 5 folds for each of 21 candidates, totalling 105 fits
# ----------
# iter: 2
# n_candidates: 11
# n_resources: 400
# Fitting 5 folds for each of 11 candidates, totalling 55 fits
# ----------
# iter: 3
# n_candidates: 6
# n_resources: 800
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
#               max_leaves=None, min_chid_weight=10, min_child_weight=None,
#               missing=nan, monotone_constraints=None, multi_strategy=None,
#               n_estimators=None, n_jobs=None, ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'min_chid_weight': 10}
# model.score : 0.9527777777777777
# best_score : 0.9285298742138366
# 걸린 시간 : 98.54 초