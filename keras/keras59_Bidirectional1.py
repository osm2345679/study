# 54-1 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import datetime


#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # (10,)

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]]) # (7,3)

y = np.array([4,5,6,7,8,9,10])  # (7,)

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)    # (7,3,1)
print(x.shape)  # (7,3,1)

#2. 모델 구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(units=10), input_shape=(3,1)))   # 행 무시 열 우선 : (7,3,1) -> (3,1)
# Wrapper class임.

# model.add(SimpleRNN(10, input_shape=(3,1)))   # 위와 같음 
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               240       
#  l)                                                              
                                                                 
#  dense (Dense)               (None, 20)                420       
                                                                 
#  dense_1 (Dense)             (None, 20)                420       
                                                                 
#  dense_2 (Dense)             (None, 20)                420       
                                                                 
#  dense_3 (Dense)             (None, 20)                420       
                                                                 
#  dense_4 (Dense)             (None, 1)                 21        
                                                                 
# =================================================================
# Total params: 1,941
# Trainable params: 1,941
# Non-trainable params: 0
# _________________________________________________________________

# 파라미터가 2배가 된 것을 확인할 수 있음
# 10*1 + 10*10 + 10*1 = 120에서 240됨

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)

path = './_save/keras54/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k54_1_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)

learning_rate = 0.01

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.fit(
    x, y,
    epochs=3000,
    batch_size=1,
    verbose=1,
    validation_split=0.01,
    callbacks=[es, mcp, rlr]
)

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)

x_predict = np.array([8, 9, 10]).reshape(-1, 3, 1)
y_predict = model.predict(x_predict)
print("[8, 9, 10]의 결과 : ", y_predict)

# Results
# Epoch 176/3000
# 1/5 [=====>........................] - ETA: 0s - loss: 2.3495e-04Restoring model weights from the end of the best epoch: 126.
# 5/5 [==============================] - 0s 9ms/step - loss: 3.0961e-04 - val_loss: 0.5238 - lr: 0.0025
# Epoch 176: early stopping
# 1/1 [==============================] - 0s 90ms/step - loss: 0.1630
# loss :  0.16299545764923096
# 1/1 [==============================] - 0s 91ms/step
# [8, 9, 10]의 결과 :  [[9.7245245]]

# Bidirectional
# Epoch 138/3000
# 1/5 [=====>........................] - ETA: 0s - loss: 1.3303e-05Restoring model weights from the end of the best epoch: 88.
# 5/5 [==============================] - 0s 7ms/step - loss: 1.2314e-04 - val_loss: 0.5330 - lr: 0.0025
# Epoch 138: early stopping
# 1/1 [==============================] - 0s 135ms/step - loss: 0.0896
# loss :  0.08962439000606537
# 1/1 [==============================] - 0s 148ms/step
# [8, 9, 10]의 결과 :  [[9.695598]]

# validation_split 0.2 -> 0.01
# Epoch 88/3000
# 1/6 [====>.........................] - ETA: 0s - loss: 3.3755e-06Restoring model weights from the end of the best epoch: 38.
# 6/6 [==============================] - 0s 5ms/step - loss: 5.5047e-06 - val_loss: 0.0860 - lr: 0.0025
# Epoch 88: early stopping
# 1/1 [==============================] - 0s 125ms/step - loss: 0.0302
# loss :  0.030229493975639343
# 1/1 [==============================] - 0s 131ms/step
# [8, 9, 10]의 결과 :  [[10.24477]]