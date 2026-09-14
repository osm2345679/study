# 20-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = min,
    patience = 150,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=2000, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6, 4))
plt.title('보스턴 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# Results
#
# Epoch 1000/1000
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 23.1287 - val_loss: 35.5166
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 23.9389 
# loss :  23.938945770263672
#
# MinMaxScaler
# Epoch 483/2000
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 20.6260 - val_loss: 30.2503
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 23.4597
# loss :  23.459732055664062
#
# StandardScaler
# Epoch 257/2000
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 21.5468 - val_loss: 31.6762
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 22.6933 
# loss :  22.693265914916992
#
# MaxAbsScaler
# Epoch 678/2000
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 20.7993 - val_loss: 30.4088
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 22.8999 
# loss :  22.89986801147461
#
# RobustScaler
# Epoch 335/2000
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 20.9083 - val_loss: 31.2385
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - loss: 22.1879 
# loss :  22.1878681182861