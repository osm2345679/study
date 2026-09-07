# 19-1 카피

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

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=8))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=2,    # 임계값
    restore_best_weights=True,  # 디폴트는 False. False면 스탑 됐을 때의 가중치를 반환.
)
# 클래스. 선언하고 fit에서 callback 추가해서 사용.

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=500000,
                 batch_size=16,
                 verbose=1,
                 validation_split=0.25,
                 callbacks=[es]
                 )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

print("=================history================")
print(hist) # <keras.src.callbacks.history.History object at 0x0000015FFEC4B610>
print("=================history================")
print(hist.history)
# 파이썬 기초 - key-value 형태. 딕셔너리.

# {'loss': [45560.26953125, 44.246402740478516, 38.073543548583984, 19.846057891845703, 13.693187713623047,
#  21.59902572631836, 20.992708206176758, 12.894192695617676, 25.63913345336914, 126.82968139648438],
# 'val_loss': [49.485870361328125, 20.810604095458984, 14.764541625976562, 11.018330574035645, 59.73923873901367,
#  8.459661483764648, 10.467756271362305, 9.65067195892334, 10.843430519104004, 5.474674701690674]}
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
# Epoch 200/200
# 774/774 ━━━━━━━━━━━━━━━━━━━━ 1s 851us/step - loss: 0.6110 - val_loss: 0.6570
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 734us/step - loss: 0.6261
# loss :  0.6261408925056458