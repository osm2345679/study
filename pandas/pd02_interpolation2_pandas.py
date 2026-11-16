import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6,  8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan],
                     ]).T
print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#0. 결측치 확인
print(data.isnull)
print(data.isnull().sum())
print(data.info())
# <bound method DataFrame.isnull of      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN>
# x1    1
# x2    2
# x3    0
# x4    3
# dtype: int64
# <class 'pandas.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64
# dtypes: float64(4)
# memory usa

#1. 결측치 삭제
print(data.dropna())    # 디폴트는 행 삭제
print(data.dropna(axis=1))
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

#2-1. 특정값 - 평균
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000
# dtype: float64
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0

#2-2. 특정값 - 중위값
med = data.median() # 중앙값. 데이터의 갯수가 짝수면 중앙에 가까운 두 값 평균.
print(med)
data3 = data.fillna(med)
print(data3)
# x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
# dtype: float64
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

#2-3. 특정값 - 0, 42, 777, 1984, ...
data4 = data.fillna(0)
print(data4)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

#2-4. 특정값 - ffill (결측치가 마지막 값이면 주로 사용), (시계열)
data5 = data.ffill()
print(data5)    # 첫번째 값은 처리 못함
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

#2-5. 특징값 - bfill (결측치가 첫번째 값이면), (시계열)
data6 = data.bfill()
print(data6)    # 마지막 값은 처리 못함.
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

print("======================")
############### 특징 열만 ###############
means = data['x1'].mean()
print(means)
# ======================
# 6.5

med = data['x4'].median()
print(med)
# 6.0

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(med)
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0
