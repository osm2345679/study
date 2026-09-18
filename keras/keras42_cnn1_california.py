# 30-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
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

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(-1,4,2,1)
x_test = x_test.reshape(-1,4,2,1)

#2. 모델 구성
model = Sequential()
# model.add(Dense(20, input_dim=8))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1))

model.add(Conv2D(64, (2,1), input_shape=(4,2,1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,1), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())    # 
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(1))  # (10,)

model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 20,    # 임계값
    restore_best_weights = True,  # 디폴트는 False. False면 스탑 됐을 때의 가중치를 반환.,
    verbose=1
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath = path + 'keras30_mcp1.keras',
    verbose=1
)


start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=5000, batch_size=16, verbose=1, validation_split=0.25, callbacks=[es])

# Epoch 1: val_loss improved from None to 0.91376, saving model to ./_save/keras30/keras30_mcp1.keras
# Epoch 1: finished saving model to ./_save/keras30/keras30_mcp1.keras
# ... Epoch 107: val_loss did not improve from 0.31203
# ... Epoch 108: early stopping
# Restoring model weights from the end of the best epoch: 88.
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 671us/step - loss: 0.3069
# loss :  0.3068580627441406
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 671us/step - loss: 0.3069

end_time = time.time()

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
# Epoch 89/5000
# 747/774 [===========================>..] - ETA: 0s - loss: 0.8772Restoring model weights from the end of the best epoch: 69.
# 774/774 [==============================] - 1s 2ms/step - loss: 0.8816 - val_loss: 0.7799
# Epoch 89: early stopping
# 129/129 [==============================] - 0s 1ms/step - loss: 0.7513
# loss :  0.7512954473495483