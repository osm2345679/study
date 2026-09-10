# 20-5 카피
# 데이터셋 : https://www.kaggle.com/competitions/bike-sharing-demand/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_log_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
path="./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.info())
# <class 'pandas.DataFrame'>
# Index: 10886 entries, 2011-01-01 00:00:00 to 2012-12-19 23:00:00
# Data columns (total 11 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   season      10886 non-null  int64  
#  1   holiday     10886 non-null  int64  
#  2   workingday  10886 non-null  int64  
#  3   weather     10886 non-null  int64  
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64  
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64  
#  9   registered  10886 non-null  int64  
#  10  count       10886 non-null  int64  
# dtypes: float64(3), int64(8)
# memory usage: 1020.6+ KB
# None
print(test_csv.info())
# <class 'pandas.DataFrame'>
# Index: 6493 entries, 2011-01-20 00:00:00 to 2012-12-31 23:00:00
# Data columns (total 8 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   season      6493 non-null   int64  
#  1   holiday     6493 non-null   int64  
#  2   workingday  6493 non-null   int64  
#  3   weather     6493 non-null   int64  
#  4   temp        6493 non-null   float64
#  5   atemp       6493 non-null   float64
#  6   humidity    6493 non-null   int64  
#  7   windspeed   6493 non-null   float64
# dtypes: float64(3), int64(5)
# memory usage: 456.5+ KB
# None
print(submission.info())
# <class 'pandas.DataFrame'>
# Index: 6493 entries, 2011-01-20 00:00:00 to 2012-12-31 23:00:00
# Data columns (total 1 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   count   6493 non-null   int64
# dtypes: int64(1)
# memory usage: 101.5+ KB
# None

print(train_csv.describe())
#              season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000

############ 결측치 확인 #################
print(train_csv.isna().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
# dtype: int64
print(test_csv.isna().sum())
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# dtype: int64

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)    # ...[10886 rows x 8 columns]
y = train_csv['count']
print(y)    # ... Name: count, Length: 10886, dtype: int64

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = min,
    patience = 100,
    restore_best_weights = True
)
hist = model.fit(x_train, y_train, epochs=3000, batch_size=32, verbose=1, validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("rmse : ", rmse)

# 제출
y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + "submit/" + "submit_0910_1741.csv")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9,9))
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='purple', label='val_loss')
plt.legend(loc='upper left')
plt.title('Kaggle 자전거 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.grid()
plt.show()

# Results
# 
# Epoch 2000/2000
# 205/205 ━━━━━━━━━━━━━━━━━━━━ 0s 991us/step - loss: 21460.0293 - val_loss: 21418.9297
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 698us/step - loss: 21497.1582
# loss :  21497.158203125
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 841us/step
# r2 :  0.3487076759338379
# mse :  21497.162109375
# rmse :  146.61910554008642
# 203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 362us/step
# 
# MinMaxScaler
