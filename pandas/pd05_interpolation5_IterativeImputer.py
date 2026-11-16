#4 카피
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6,  8, 10, 8],
                     [2, 4, np.nan, 8, np.nan, 4],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan, 8],
                     ]).T
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

imputer = IterativeImputer(estimator=xgb, max_iter=10, tol = 0.001) # 디폴트

data2 = imputer.fit_transform(data)
print(data2)

print("실제 반복 횟수 :", imputer.n_iter_)
# 실제 반복 횟수 : 3
