import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
import time

"""
다중 분류 프로세스
1. 데이터 받으면 OnehotEncoding 하기 - to_categorical
2. 아웃풋 레이어에서 activation은 - softmax
3. 컴파일에서 loss는 - categorical_crossentropy
"""

# 1. 데이터
datasets = load_iris()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([50, 50, 50]))

'''
One-hot encoding
[0, 0, 1, 1, 2] #(5,)
->
[[1,0,0]
 [1,0,0]
 [0,1,0]
 [0,1,0]
 [0,0,1]]   #(5,3)
'''

################### One-hot encoding method 1. keras.utils - to_categorical() ######################

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

################### One-hot encoding method 2. pandas - pd.get_dummies()) ######################

# y = pd.get_dummies(y, dtype=int)

### Below codes are other related class and function ###
# y = pd.from_dummies(y)
# y = pd.Categorical(y)

################### One-hot encoding method 3. sklearn - OneHotEncoder / fit_transform()) ######################

# y = y.reshape(150, 1)
y = y.reshape(-1, 1)    # -1로 놓으면 배열 길이와 다른 차원을 통해 알아서 추론함.
# reshape 조건 - 1. 내용 유지, 2. 순서 유지 되는 경우 가능.
y = OneHotEncoder(sparse_output=False).fit_transform(y) # sparse_out : defalut는 True. True면 sparse matrix를 CSR 형식으로 리턴함.

### Below codes are other related class and function ###
# mlb = MultiLabelBinarizer()
# y = y.reshape(-1, 1)
# y = mlb.fit_transform(y)

print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=333,
    shuffle=True,
    stratify=y
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (120, 4) (30, 4) (120, 3) (30, 3)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
start_time = time.time()
model.fit(x_train, y_train,
    epochs=2000,
    batch_size = 12,
    callbacks= [es],
    validation_split=0.2
)
end_time = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1]))

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# print(x_test.shape)
# print(y_pred)
# print(y_argmax)
# print(y_argmax.shape)

# print(y_test)
# print(y_test.shape)

# print(y_met_argmax)
# print(y_met_argmax.shape)

acc = accuracy_score(y_test, y_argmax)
print("acc_score : ", acc)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")

# Results
# 
# Epoch 1360/2000
# 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - acc: 0.9896 - loss: 0.0345 - val_acc: 1.0000 - val_loss: 0.0352
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step - acc: 0.9667 - loss: 0.1047
# loss :  0.10466615110635757
# acc :  1
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
# acc_score :  0.9666666666666667
# 걸린 시간 :  73.16 초