# 데이터셋 : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1200, activation='relu'))
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
    batch_size=32,
    verbose=2,
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
