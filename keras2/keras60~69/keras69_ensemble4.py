# 69-3 카피
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import time

#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).T # (100, 2)
                        # 삼성종가      # 하이닉스

y1 = np.array(range(3001, 3101)) # (100,)
                # 화성의 화씨 온도

y2 = np.array(range(13001, 13101))  # (100,)
                # 비트코인 가격

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets, y1, y2,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# print(x1_train.shape, x1_test.shape)    # (80, 2) (20, 2)
# print(x2_train.shape, x2_test.shape)    # (80, 3) (20, 3)
# print(y_train.shape, y_test.shape)      # (80,) (20,)

#2. 모델 구성
#2-1.
input1 = Input(shape=(2))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
dense4 = Dense(40, activation='relu', name='han4')(dense3)
dense5 = Dense(50, activation='relu', name='han5')(dense4)
# model1 = Model(inputs=input1, outputs=dense5)

#2-2. 분기 1
dense41 = Dense(50, activation='relu', name='han41')(dense5)
dense42 = Dense(50, activation='relu', name='han42')(dense41)
dense43 = Dense(50, activation='relu', name='han43')(dense42)
dense44 = Dense(50, activation='relu', name='han44')(dense43)
output1 = Dense(1, name='han45')(dense44)

#2-3. 분기 2
dense51 = Dense(50, activation='relu', name='han51')(dense5)
dense52 = Dense(50, activation='relu', name='han52')(dense51)
dense53 = Dense(50, activation='relu', name='han53')(dense52)
dense54 = Dense(50, activation='relu', name='han54')(dense53)
output2 = Dense(1, name='han55')(dense54)

#2-7. model
model = Model(inputs=[input1], outputs=[output1, output2])

#3. 컴파일, 훈련
learning_rate = 0.001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)

path = './_save/keras69/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path, 'keras69_4_', date, '-', filename])

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    factor=0.5
)

model.fit(
    [x_train], [y1_train, y2_train],
    epochs=3000,
    verbose=1,
    validation_split=0.25,
    callbacks=[es, mcp, rlr]
)

#4. 평가, 예측
results = model.evaluate([x_test], [y1_test, y2_test])
print("loss : ", results)

x_pred = np.array([range(100, 106), range(400, 406)]).T    # (6, 2)

y_pred = model.predict([x_pred])
# print(type(y_pred)) # <class 'list'>
print("x_pred의 예측값 1 : ", y_pred[0])
print("x_pred의 예측값 2 : ", y_pred[1])

# Results
# Epoch 518/3000
# 1/2 [==============>...............] - ETA: 0s - loss: 7.7244 - han45_loss: 0.5277 - han55_loss: 7.1967Restoring model weights from the end of the best epoch: 468.

# Epoch 518: val_loss did not improve from 3.95381

# Epoch 518: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 2/2 [==============================] - 0s 31ms/step - loss: 6.6354 - han45_loss: 0.4746 - han55_loss: 6.1608 - val_loss: 7.4981 - val_han45_loss: 0.4799 - val_han55_loss: 7.0183 - lr: 0.0010
# Epoch 518: early stopping
# 1/1 [==============================] - 0s 15ms/step - loss: 5.5714 - han45_loss: 0.9634 - han55_loss: 4.6079
# loss :  [5.571364879608154, 0.9634330868721008, 4.607931613922119]
# 1/1 [==============================] - 0s 76ms/step
# x_pred의 예측값 1 :  [[3096.7104]
#  [3102.204 ]
#  [3107.2424]
#  [3113.7014]
#  [3120.8945]
#  [3126.1372]]
# x_pred의 예측값 2 :  [[13085.186]
#  [13104.989]
#  [13121.118]
#  [13143.174]
#  [13168.28 ]
#  [13188.758]]