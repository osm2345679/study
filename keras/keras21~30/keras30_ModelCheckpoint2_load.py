# 30-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

path = './_save/keras30/'


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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = load_model(path + 'keras30_mcp1.keras')

# model = Sequential()
# model.add(Dense(8, input_dim=8))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

model.summary()

# exit()

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(
#     monitor='val_loss',
#     mode = 'min',
#     patience = 20,    # 임계값
#     restore_best_weights = True,  # 디폴트는 False. False면 스탑 됐을 때의 가중치를 반환.,
#     verbose=1
# )

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath = path + 'keras30_mcp1.keras',
#     verbose=1
# )


# start_time = time.time()
# hist = model.fit(x_train, y_train,
#                  epochs=5000, batch_size=16, verbose=1, validation_split=0.25, callbacks=[es, mcp])

# Epoch 1: val_loss improved from None to 0.91376, saving model to ./_save/keras30/keras30_mcp1.keras
# Epoch 1: finished saving model to ./_save/keras30/keras30_mcp1.keras
# ... Epoch 107: val_loss did not improve from 0.31203
# ... Epoch 108: early stopping
# Restoring model weights from the end of the best epoch: 88.
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 671us/step - loss: 0.3069
# loss :  0.3068580627441406
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 671us/step - loss: 0.3069

# end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# print("걸린 시간 : ", round(end_time - start_time, 2), "초")

# plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 깨짐 방지
# plt.rcParams['axes.unicode_minus'] = False  # minus 기호 깨짐 방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][2:], color='red', label='loss')   # y값만 넣으면 시간 순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
# plt.legend(loc='upper right') # loc : location. 디폴트 빈 공간 아무데나. 우상단에 표시.
# plt.title('캘리포니아 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss/val_loss')
# plt.grid()  # 격자 표시 추가
# plt.show()

# Results
# Epoch 200/200
# 774/774 ━━━━━━━━━━━━━━━━━━━━ 1s 851us/step - loss: 0.6110 - val_loss: 0.6570
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 734us/step - loss: 0.6261
# loss :  0.6261408925056458
#
# MinMaxScaler
# Epoch 1000/1000
# 774/774 ━━━━━━━━━━━━━━━━━━━━ 1s 779us/step - loss: 0.5096 - val_loss: 0.5507
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 743us/step - loss: 0.5517
# loss :  0.551708996295929
# 걸린 시간 :  686.83 초

# StandardScaler
# Epoch 500/100000
# 774/774 ━━━━━━━━━━━━━━━━━━━━ 1s 887us/step - loss: 0.2825 - val_loss: 0.3234
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 553us/step - loss: 0.3355
# loss :  0.33548685908317566
# 걸린 시간 :  455.82 초

# MinAbsScaler
# Epoch 605/100000
# 774/774 ━━━━━━━━━━━━━━━━━━━━ 1s 827us/step - loss: 0.3455 - val_loss: 0.3859
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 721us/step - loss: 0.3577
# loss :  0.357735812664032
# 걸린 시간 :  393.37 초

# RobustScaler
# Epoch 327/100000
# 774/774 ━━━━━━━━━━━━━━━━━━━━ 1s 898us/step - loss: 0.2912 - val_loss: 0.3302
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 701us/step - loss: 0.3039
# loss :  0.3039208650588989
# 걸린 시간 :  218.39 초