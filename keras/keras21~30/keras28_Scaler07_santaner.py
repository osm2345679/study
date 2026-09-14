# 22 카피
# 데이터셋 : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf

# 1. 내부 연산(Matrix Multiplication 등)에 사용할 스레드 수
tf.config.threading.set_intra_op_parallelism_threads(16)

# 2. 독립적인 연산들을 병렬로 처리할 스레드 수
tf.config.threading.set_inter_op_parallelism_threads(16)


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

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=70,
    restore_best_weights=True
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

submit = model.predict(test_csv)
submit = np.round(submit)   # 이진분류이므로 라운드 처리
submission_csv['target'] = submit
submission_csv.to_csv(path + "submit/" + "0908_1642.csv")

# Results
# 
# Epoch 37/3000
# 3750/3750 ━━━━━━━━━━━━━━━━━━━━ 10s 3ms/step - acc: 0.9167 - loss: 0.2261 - val_acc: 0.9112 - val_loss: 0.2546
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 909us/step - acc: 0.9094 - loss: 0.2455
# loss :  [0.24553681910037994, 0.909375011920929]
# 걸린 시간 :  352.1 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 739us/step 
# acc_score :  0.909375
# 6250/6250 ━━━━━━━━━━━━━━━━━━━━ 4s 700us/step 
#
# MinMaxScaler
# Epoch 41/3000
# 3750/3750 ━━━━━━━━━━━━━━━━━━━━ 9s 2ms/step - acc: 0.9145 - loss: 0.2322 - val_acc: 0.9147 - val_loss: 0.2351
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 918us/step - acc: 0.9128 - loss: 0.2336
# loss :  [0.2335789054632187, 0.9127500057220459]
# 걸린 시간 :  387.88 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 744us/step
# acc_score :  0.91275
# 6250/6250 ━━━━━━━━━━━━━━━━━━━━ 5s 758us/step 
#
# StandardScaler
# Epoch 71/3000
# 938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 5ms/step - acc: 0.9983 - loss: 0.0056 - val_acc: 0.8916 - val_loss: 1.2330
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - acc: 0.9110 - loss: 0.2417  
# loss :  [0.2416592240333557, 0.9110000133514404]
# 걸린 시간 :  312.12 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 725us/step
# acc_score :  0.911
# 6250/6250 ━━━━━━━━━━━━━━━━━━━━ 5s 730us/step
#
# MaxAbsScaler
# Epoch 80/3000
# 938/938 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step - acc: 0.9775 - loss: 0.0707 - val_acc: 0.8847 - val_loss: 0.7258
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - acc: 0.9115 - loss: 0.2412
# loss :  [0.2412102222442627, 0.9115250110626221]
# 걸린 시간 :  353.31 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 865us/step
# acc_score :  0.911525
# 6250/6250 ━━━━━━━━━━━━━━━━━━━━ 6s 892us/step 
#
# RobustScaler
# Epoch 72/3000
# 938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - acc: 0.9977 - loss: 0.0064 - val_acc: 0.8845 - val_loss: 1.3126
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 944us/step - acc: 0.9105 - loss: 0.2420
# loss :  [0.2420067936182022, 0.9105499982833862]
# 걸린 시간 :  308.49 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 741us/step
# acc_score :  0.91055
# 6250/6250 ━━━━━━━━━━━━━━━━━━━━ 5s 868us/step 