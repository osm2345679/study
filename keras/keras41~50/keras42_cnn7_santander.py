# 22 카피
# 데이터셋 : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import datetime

#1. 데이터
path = 'c:/study/_data/kaggle_santander/'
# path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print(train_csv.shape)  # (200000, 201)

print(train_csv.isna().sum())
# target     0
# var_0      0
# var_1      0
# var_2      0
# var_3      0
#           ..
# var_195    0
# var_196    0
# var_197    0
# var_198    0
# var_199    0
# Length: 201, dtype: int64

print(train_csv.info())
# <class 'pandas.DataFrame'>
# Index: 200000 entries, train_0 to train_199999
# Columns: 201 entries, target to var_199
# dtypes: float64(200), int64(1)
# memory usage: 308.2+ MB
# None

print(test_csv.isna().sum())
print(test_csv.info())

x = train_csv.drop('target', axis=1)
print(x.shape)  # (200000, 200)
print(type(x))  # <class 'pandas.DataFrame'> 
y = train_csv['target']
print(y.shape)  # (200000,)
print(type(y))  # <class 'pandas.Series'>

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(np.unique(x, return_counts=True))
# (array([-90.2525, -83.1075, -82.2573, ...,  70.272 ,  70.8691,  74.0321], shape=(828834,)), array([1, 1, 1, ..., 1, 1, 1], shape=(828834,)))
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098]))

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,25,8,1)
x_test = x_test.reshape(-1,25,8,1)

#2. 모델 구성
model = Sequential()
# model.add(Dense(200, input_dim=200, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(600, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

model.add(Conv2D(64, (2,1), input_shape=(25,8,1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,1), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())    # 
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(1, activation='sigmoid'))  # (10,)

#3. 컴파일, 훈련
path = './_save/keras31/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k31_7_", date, "-", filename])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=70,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)
start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=128,
    verbose=1,
    callbacks=[es],
    validation_split=0.25
)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)

acc_score = accuracy_score(y_test, np.round(y_pred))
print("acc_score : ", acc_score)

plt.figure(figsize=(9,6))
plt.plot(hist.history['acc'], c='red', label='acc')
plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Acc/Val_acc/Loss/Val_loss')
plt.legend(loc='upper right')
plt.title('Kaggle Santander Results')
plt.grid()
plt.show()

# submit = model.predict(test_csv)
# submit = np.round(submit)   # 이진분류이므로 라운드 처리
# submission_csv['target'] = submit
# submission_csv.to_csv(path + "submit/" + "0908_1642.csv")

# Results
# Epoch 375/3000
# 938/938 ━━━━━━━━━━━━━━━━━━━━ 7s 7ms/step - acc: 0.9006 - loss: 0.2906 - val_acc: 0.8991 - val_loss: 0.2937
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - acc: 0.9008 - loss: 0.2907  
# loss :  [0.29072949290275574, 0.9007750153541565]
# 걸린 시간 :  2673.16 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 891us/step
# acc_score :  0.900775