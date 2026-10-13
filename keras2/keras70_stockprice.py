# 삼성전자 내일(10/13) 시가 맞추기
# 앙상블로 구성
# LSTM, Conv1D 사용

# 할 일
#1. 데이터
# - 데이터 불러오기
# - 사용할 칼럼들(x) 결정하기
# - x1, x2 만들기
# - y 만들기
# - split_x로 분할하기
# - train_test_split
# - scaling

#2. 모델 구성
# - 2-1, 2-2 LSTM, Conv1D layer 넣고 모델 구성
# - 2-3 concatenate로 merge하고 output까지 모델 구성
# - output은 시가로 1

#3. 컴파일, 훈련 & #4. 평가, 예측
# - 회귀로 구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Conv1D, Dropout, Dense, GlobalAveragePooling1D, Flatten, Input, Reshape, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import pandas as pd
import numpy as np
import datetime
import time

path = './_data/ensemble/'
size = 10

#1. 데이터
# - 데이터 불러오기
samsung = pd.read_csv(path + '삼성전자261012.csv', encoding='euc-kr', index_col=0, thousands=',')
# ValueError: could not convert string to float: '199,900' 에러 나서 thousands=',' 추가함. 콤마 제거
# x1 = x1.loc[:, ['시가', '종가', '거래량', '개인', '기관', '외인(수량)', '외국계', '프로그램']].str.replace(',', '').astype(float)
# x1 = x1['시가'].str.replace(',', '')
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 0: invalid start byte 에러나서 euc-kr로 인코딩

# print(samsung.info())
# <class 'pandas.core.frame.DataFrame'>
# Index: 1120 entries, 2026/10/12 to 2022/03/14
# Data columns (total 16 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   시가          1120 non-null   object 
#  1   고가          1120 non-null   object 
#  2   저가          1120 non-null   object 
#  3   종가          1120 non-null   object 
#  4   대비          1120 non-null   object 
#  5   Unnamed: 6  1120 non-null   object 
#  6   등락률         1120 non-null   float64
#  7   거래량         1120 non-null   object 
#  8   금액(백만)      1120 non-null   object 
#  9   신용비         1120 non-null   float64
#  10  개인          1120 non-null   object 
#  11  기관          1120 non-null   object 
#  12  외인(수량)      1120 non-null   object 
#  13  외국계         1120 non-null   object 
#  14  프로그램        1120 non-null   object 
#  15  외인비         1120 non-null   float64
# dtypes: float64(3), object(13)
# memory usage: 148.8+ KB
# None

# print(samsung.describe())
#                등락률          신용비          외인비
# count  1120.000000  1120.000000  1120.000000
# mean      0.155696     0.190241    51.411250
# std       2.712894     0.101787     2.420831
# min     -13.390000     0.000000    46.350000
# 25%      -1.090000     0.110000    49.820000
# 50%       0.000000     0.170000    50.900000
# 75%       1.215000     0.260000    53.042500
# max      26.810000     0.430000    56.550000

# print(samsung.shape) # (1120, 17)

# print(samsung[:3])
#                  시가       고가       저가       종가 대비 Unnamed: 6   등락률         거래량     금액(백만)   신용비         개인          기관      외인(수량)         외국계        프로그램    외인비
# 일자                                                                                                                                                               
# 2026/10/12  260,000  262,500  254,000  255,000  ▼     -7,000 -2.67   5,119,682  1,321,121  0.00          0           0           0    -130,343    -120,410  46.35
# 2026/10/08  269,500  270,000  262,000  263,000  ▼     -5,500 -2.05  21,259,056  5,617,135  0.36  5,259,380  -4,087,636  -1,755,574  -1,610,540  -3,003,100  46.35
# 2026/10/07  269,500  279,500  268,000  269,000  ▼     -3,000 -1.10  16,395,504  4,471,785  0.36    957,691    -819,541     372,168   1,533,227    -330,771  46.38

hynix = pd.read_csv(path + '하이닉스261012.csv', encoding='euc-kr', index_col=0, thousands=',')
amore = pd.read_csv(path + '아모레261012.csv', encoding='euc-kr', index_col=0, thousands=',')

# - 사용할 칼럼들(x) 결정하기
# 시가, 종가, 거래량, 개인, 기관, 외인(수량), 외국계, 프로그램

# - x1, x2 만들기
x1 = hynix.loc[:, ['시가', '종가', '거래량', '개인', '기관', '외인(수량)', '외국계', '프로그램']]
# print(x1.shape) # (1300, 8)
# print(x1[:5])
#                    시가         종가        거래량        개인        기관    외인(수량)       외국계      프로그램
# 일자                                                                                           
# 2026/10/12  1,640,000  1,606,000    961,884         0         0         0  -133,444  -123,956
# 2026/10/08  1,722,000  1,686,000  3,203,231   130,172   -22,595  -545,114  -403,266    36,046
# 2026/10/07  1,728,000  1,715,000  2,618,253   498,962  -107,621  -849,668  -374,358  -401,805
# 2026/10/06  1,835,000  1,784,000  2,811,364   534,915  -145,641  -416,790  -717,928  -460,660
# 2026/10/02  1,830,000  1,842,000  2,032,332  -452,221    76,049   -39,024   -11,562   -72,143

x2 = amore.loc[:, ['시가', '종가', '거래량', '개인', '기관', '외인(수량)', '외국계', '프로그램']]
# print(x2.shape) # (2060, 8)
# print(x2[:5])
#                  시가       종가      거래량       개인       기관   외인(수량)      외국계     프로그램
# 일자                                                                                
# 2026/10/12  133,700  131,000   52,798        0        0        0    1,428   11,254
# 2026/10/08  136,400  133,900  199,873   12,454  -37,314   17,867   17,616   43,755
# 2026/10/07  134,000  138,300  242,450  -67,242   21,332   22,395   27,619   62,962
# 2026/10/06  133,400  133,000  244,546   17,039   15,442  -27,652  -56,043  -10,517
# 2026/10/02  136,000  132,500  193,414   25,852   -9,504  -31,381  -16,505     -384

# - y 만들기
y = samsung['시가']
# print(y.shape)  # (1120,)
# print(y[:5])
# 일자
# 2026/10/12    260,000
# 2026/10/08    269,500
# 2026/10/07    269,500
# 2026/10/06    278,500
# 2026/10/02    273,500
# Name: 시가, dtype: object

# 전처리에서 추가로 해야 할 것들
# - 데이터 간 행 갯수가 안 맞음. 날짜 최근 순이라 아랫부분 자르면 됨
# - 분할 전 미리 자르기
#   : 오늘의 hynix, amore로 내일의 samsung 예측
#   오늘(10/12) hynix, amore 데이터는 y 데이터가 없음. 예측 시 넣을 값으로 이걸로 내일 samsung 예측해야 함
#   어제의 hynix, amore와 오늘의 samsung 시가는 써도 되는가. 써도 될듯? 오늘 시가까지 있는 데이터셋이니까.
#   그러면 x1, x2에서는 맨 끝만 자르고, y에서는 맨 위만 자르면 될듯.
#   -> 오늘 말고 시계열 데이터로 분할해서 쓰기 위해서 10일의 x1, x2로 그리고 오늘치 데이터는 결측치 많으므로 이틀 뒤의 y 예측으로 변경
#   그리고 날짜 최근 순이라 반대로 잘라야 함.
#   예측에 쓸 데이터가 두 번째 행부터이므로 x1, x2는 맨 위에서 11개 잘라야 함.
#   y는 이틀 뒤 예측이므로 맨 아래에서 11개 자르고 맨 위에서 9개 잘라야 x1, x2의 마지막 예측의 결과값에 맞을듯

# - 데이터 행 갯수 맞추기
# print(x1[1115:1120])
# print(x2[1115:1120])
# print(y[1115:])
# 일자                                                                                             
# 2022/03/18  124,000  124,000  4,493,159    -244,438    40,754    157,806    527,881  -1,003,518
# 2022/03/17  121,000  124,000  7,219,813  -1,739,636  -290,046  2,096,503  1,896,776   1,349,621
# 2022/03/16  115,000  116,500  3,739,977    -320,925   129,509    122,915   -162,417      48,773
# 2022/03/15  114,500  112,500  3,578,256     925,807  -100,777   -654,064   -653,102    -656,154
# 2022/03/14  115,500  116,000  3,172,353     249,106   -20,692   -199,753   -326,228     115,129
#                  시가       종가      거래량       개인        기관    외인(수량)       외국계      프로그램
# 일자                                                                                    
# 2022/03/18  154,500  151,000  606,826  338,073  -100,519  -227,501  -229,021   -21,933
# 2022/03/17  158,000  154,000  772,840  321,371    29,688  -362,375  -228,596  -163,864
# 2022/03/16  155,500  155,000  344,802   46,145    14,548   -49,539   -19,488   -74,363
# 2022/03/15  152,000  153,500  365,015   95,269   -14,799  -107,972    13,404  -119,023
# 2022/03/14  163,500  154,000  703,326  294,867   -69,259  -233,749   -78,897  -214,018
# 일자
# 2022/03/18    70,600
# 2022/03/17    71,200
# 2022/03/16    70,200
# 2022/03/15    69,800
# 2022/03/14    70,000
# Name: 시가, dtype: object

x1 = x1[:1120]
x2 = x2[:1120]
# print(x1.shape, x2.shape, y.shape)  # (1120, 8) (1120, 8) (1120,)

# - 예측할 x1_pred, x2_pred 미리 만드는 게 나을 것 같아서 추가
x1_pred = x1[1 : (size + 1)]
x2_pred = x2[1 : (size + 1)]
print(x1_pred.shape)    # (10, 8)
print(x2_pred.shape)    # (10, 8)

# - 분할 전 미리 자르기
x1 = x1[(size + 1):]
x2 = x2[(size + 1):]
# y = y[(size - 1) : -(size + 1)]
y = y[(size - 1) : -(size + 1)]
# print(x1.shape, x2.shape, y.shape)  # (1109, 8) (1109, 8) (1100,)

# - split_x로 분할하기
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(x1, size)
x2 = split_x(x2, size)
# print(x1.shape, x2.shape, y.shape)  # (1100, 10, 8) (1100, 10, 8) (1100,)

# - train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y,
    test_size=0.2,
    random_state=42
)

# - scaling
# scaling 위해서 2차원으로 reshape
# print(x1_train.shape, x1_test.shape)    # (880, 10, 8) (220, 10, 8)
# print(x2_train.shape, x2_test.shape)    # (880, 10, 8) (220, 10, 8)

x1_train = x1_train.reshape(-1, 8)
x1_test = x1_test.reshape(-1, 8)

x2_train = x2_train.reshape(-1, 8)
x2_test = x2_test.reshape(-1, 8)

x1_scaler = RobustScaler()
x1_train = x1_scaler.fit_transform(x1_train, x1_train)
x1_test = x1_scaler.transform(x1_test)

x2_scaler = RobustScaler()
x2_train = x2_scaler.fit_transform(x2_train, x2_train)
x2_test = x2_scaler.transform(x2_test)

# scaling 끝내고 원래대로 reshape
x1_train = x1_train.reshape(-1, 10, 8)
x1_test = x1_test.reshape(-1, 10, 8)

x2_train = x2_train.reshape(-1, 10, 8)
x2_test = x2_test.reshape(-1, 10, 8)

# print(x1_train.shape, x1_test.shape)    # (880, 10, 8) (220, 10, 8)
# print(x2_train.shape, x2_test.shape)    # (880, 10, 8) (220, 10, 8)

# - x1_pred, x2_pred도 split, scaling 처리

#  UserWarning: X has feature names, but MaxAbsScaler was fitted without feature names warnings.warn( 오류 나서 추가
#  구글링 해보니 pandas scaling해서 나는 오류인듯
x1_pred = np.array(x1_pred)
x2_pred = np.array(x2_pred)

#   : scaling 먼저하는 게 편하고 문제될 거 없어서 scaling 먼저 함
x1_pred = x1_scaler.transform(x1_pred)
x2_pred = x2_scaler.transform(x2_pred)

# split
x1_pred = split_x(x1_pred, size)
x2_pred = split_x(x2_pred, size)
# print(x1_pred.shape, x2_pred.shape) # (1, 10, 8) (1, 10, 8)

np_path = path
np.save(np_path + 'keras70_x1_train.npy', arr=x1_train)
np.save(np_path + 'keras70_x1_test.npy', arr=x1_test)
np.save(np_path + 'keras70_x2_train.npy', arr=x2_train)
np.save(np_path + 'keras70_x2_test.npy', arr=x2_test)
np.save(np_path + 'keras70_y_train.npy', arr=y_train)
np.save(np_path + 'keras70_y_test.npy', arr=y_test)
np.save(np_path + 'keras70_x1_pred.npy', arr=x1_pred)
np.save(np_path + 'keras70_x2_pred.npy', arr=x2_pred)

#2. 모델 구성
#2-1.
input1 = Input(shape=(10, 8))
lstm1 = LSTM(400)(input1)   # (None, 100)
reshape1 = Reshape((20, 20))(lstm1) # (None, 10, 10)
conv1d1 = Conv1D(20, 3)(reshape1)    # (None, 6, 10)
conv1d2 = Conv1D(20, 3)(conv1d1)    # (None, 4, 10)
conv1d3 = Conv1D(20, 3)(conv1d2)    # (None, 2, 10)
gap1 = GlobalAveragePooling1D()(conv1d3)
dense1 = Dense(200, activation='relu')(gap1)

#2-2.
input11 = Input(shape=(10, 8))
lstm11 = LSTM(400)(input11)   # (None, 100)
reshape11 = Reshape((20, 20))(lstm11) # (None, 20, 20)
conv1d11 = Conv1D(20, 3)(reshape11)    # (None, 18, 10)
conv1d12 = Conv1D(20, 3)(conv1d11)    # (None, 4, 10)
conv1d13 = Conv1D(20, 3)(conv1d12)    # (None, 2, 10)
gap11 = GlobalAveragePooling1D()(conv1d13)
dense11 = Dense(200, activation='relu')(gap1)

#2-3.
merge1 = Concatenate()([dense1, dense11])
merge2 = Dense(200, activation='relu')(merge1)
merge3 = Dense(200, activation='relu')(merge2)
merge4 = Dense(200, activation='relu')(merge3)
output1 = Dense(1)(merge4)

model = Model(inputs=[input1, input11], outputs=output1)
model.summary()

#3. 컴파일, 훈련
learning_rate = 0.001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)

path = './_save/keras70/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path, 'keras70_', date, '-', filename])

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.5
)

model.fit(
    x=[x1_train, x2_train], y=y_train,
    epochs=3000,
    verbose=1,
    validation_split=0.25,
    callbacks=[es, mcp, rlr]
)

#4. 평가, 예측
results = model.evaluate(x=[x1_test, x2_test], y=y_test)
print("loss : ", results)

y_pred = model.predict(x=[x1_pred, x2_pred])
print("2026-10-13 삼성전자 시가 예측  : ", y_pred, "원")

# Results
# Epoch 195/3000
# 13/21 [=================>............] - ETA: 0s - loss: 2548504.2500Restoring model weights from the end of the best epoch: 145.

# Epoch 195: val_loss did not improve from 36891960.00000

# Epoch 195: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 21/21 [==============================] - 0s 6ms/step - loss: 2345059.0000 - val_loss: 54315376.0000 - lr: 0.0010
# Epoch 195: early stopping
# 7/7 [==============================] - 0s 2ms/step - loss: 64233860.0000
# loss :  64233860.0
# 1/1 [==============================] - 0s 246ms/step
# 2026-10-13 삼성전자 시가 예측  :  [[262099.44]] 원