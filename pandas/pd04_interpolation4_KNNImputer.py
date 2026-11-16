#3-3-2 카피
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6,  8, 10, 8],
                     [2, 4, np.nan, 8, np.nan, 4],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan, 8],
                     ]).T
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

imputer = KNNImputer(n_neighbors=5) # defalut
data1 = imputer.fit_transform(data)
print(data1)
# [[ 2.          2.          2.          6.66666667]
#  [ 6.8         4.          4.          4.        ]
#  [ 6.          4.5         6.          6.66666667]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.5        10.          6.66666667]
#  [ 8.          4.         12.          8.        ]]