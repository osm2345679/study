# 27 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

"""
MinMaxScaler

원값 - Min
----------
Max - Min
"""

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=8))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=1, validation_split=0.25)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

print("걸린 시간 : ", round(end_time - start_time, 2), "초")
print("=================loss================")
print(hist.history['loss'])
print("=================val_loss================")
print(hist.history['val_loss'])

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False  # minus 기호 깨짐 방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:], color='red', label='loss')   # y값만 넣으면 시간 순으로 그려줌.
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
plt.legend(loc='upper right') # loc : location. 디폴트 빈 공간 아무데나. 우상단에 표시.
plt.title('캘리포니아 Loss')
plt.xlabel('epochs')
plt.ylabel('loss/val_loss')
plt.grid()  # 격자 표시 추가
plt.show()

# Results
# Epoch 1000/1000
# 774/774 ━━━━━━━━━━━━━━━━━━━━ 1s 779us/step - loss: 0.5096 - val_loss: 0.5507
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 743us/step - loss: 0.5517
# loss :  0.551708996295929
# 걸린 시간 :  686.83 초