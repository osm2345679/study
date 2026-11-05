# 7 카피

from sklearn.datasets import load_digits, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

parameters = [
    {"n_estimators" : [100, 500, 100], "max_depth" : [6, 10, 12], "learning_rate" : [0.1, 0.01, 0.001, 0.0001]},  # 36번
    {"max_depth" : [6, 8, 10, 12], "learning_rate" : [0.1, 0.01, 0.001, 0.0001]},  # 16번
    {'min_chid_weight' : [2, 3, 5, 10, 12], "learning_rate" : [0.1, 0.01, 0.001, 0.0001]} # 20번
]   # 72번

#2. 모델 구성
xgb = XGBRegressor()
model = GridSearchCV(
    xgb,
    parameters,
    cv=kfold,
    verbose=1,
    n_jobs=-1
)

#3. 컴파일, 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)

#4. 평가, 예측
print("model.score :", model.score(x_test, y_test))
print("best_score :", model.best_score_)    # train 데이터로만 나온 결과. 과적합 되어 있을 확률 높음.

print("걸린 시간 :", round(end-start, 2), "초")

# Results
# Fitting 5 folds for each of 72 candidates, totalling 360 fits
# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              feature_weights=None, gamma=None, grow_policy=None,
#              importance_type=None, interaction_constraints=None,
#              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#              max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
#              max_leaves=None, min_child_weight=None, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=100,
#              n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
# model.score : 0.35947948911819627
# best_score : 0.31553823772520556
# 걸린 시간 : 12.81 초