import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import time
import matplotlib.pyplot as plt

'''
회귀코드와의 차이
1. 요소 균형 확인 - unique
2. 훈련,테스트 계층화해서 분할 - stratify
*** 3. 레이어 마지막 activation function 고정 - sigmoid ***
*** 4. 컴파일에서 loss function 고정(이진분류에서) - binary_crossentropy ***
'''

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets['target']  # 이것도 가능.

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>
# 파이썬 기초 - type() 타입 반환

print(y)
# 0과 1의 갯수가 몇 개인지 찾아보기 - numpy
print(np.unique(y)) # [0 1]. np.unique() : 들어있는 요소들 확인
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357])). return_counts : 요소별 개수 확인

# 0과 1의 갯수가 몇 개인지 찾아보기 - pandas
print(pd.DataFrame(y).value_counts())
# 0
# 1    357
# 0    212
# Name: count, dtype: int64
print(pd.Series(y).value_counts()) # 얘도 가능. 결과는 동일.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # stratify : 계층화하다. 원데이터셋의 비율 유지하기 위함.
)

print(np.unique(y_train, return_counts=True))   # (array([0, 1]), array([170, 285]))
print(np.unique(y_test, return_counts=True))    # (array([0, 1]), array([42, 72]))

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 끝은 무조건 sigmoid로.

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', # loss는 무조건 binary_crossentropy로 고정(이진 분류에서)
              optimizer='adam',
              # metrics=['accuracy']  # 보조지표 accuracy
              metrics=['acc'] # 이것도 가능
              )
es = EarlyStopping(
    monitor='val_loss',
    mode=min,
    patience=50,
    restore_best_weights=True
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=2000, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0], 4)) # loss :  0.2158
print("acc : ", round(loss[1], 4))  # acc :  0.9298

y_pred = model.predict(x_test)
print(y_pred)   # [[2.19274860e-16] [1.00000000e+00] ... [9.97549653e-01]]. sigmoid쓰면 0, 1 사이 값 되고 반올림에서 실제 분류 처리함.

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.plot(hist.history['acc'], c='blue', label='accuracy')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.title('Breast Cancer Loss')
plt.grid()
plt.show()

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, np.round(y_pred))    # accuracy_score를 위해 y_pred round 처리
print("acc_score :", acc_score) # acc_score : 0.9210526315789473