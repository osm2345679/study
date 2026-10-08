# 69-2 카피
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import time

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T # (100, 2)
                        # 삼성종가      # 하이닉스
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() # (100, 3)
                        # 원유              # 환율          # 금시세
x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).T # (100, 4)

y1 = np.array(range(3001, 3101)) # (100,)
                # 화성의 화씨 온도

y2 = np.array(range(13001, 13101))  # (100,)
                # 비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2,
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

#2-2.
input21 = Input(shape=(3))
dense21 = Dense(50, activation='relu', name='han21')(input21)
dense22 = Dense(40, activation='relu', name='han22')(dense21)
dense23 = Dense(30, activation='relu', name='han23')(dense22)
dense24 = Dense(20, activation='relu', name='han24')(dense23)
dense25 = Dense(10, activation='relu', name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=dense25)

#2-3.
input31 = Input(shape=(4))
dense31 = Dense(50, activation='relu', name='han31')(input31)
dense32 = Dense(50, activation='relu', name='han32')(dense31)
dense33 = Dense(50, activation='relu', name='han33')(dense32)
dense34 = Dense(50, activation='relu', name='han34')(dense33)
dense35 = Dense(50, activation='relu', name='han35')(dense34)
# model2 = Model(inputs=input21, outputs=dense25)

#2-4. merge
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([dense3, dense23], name='mg1')   # 아래와 동일
merge1 = Concatenate(name='mg1')([dense5, dense25, dense35])
merge2 = Dense(50, name='mg2')(merge1)
merge3 = Dense(50, name='mg3')(merge2)
# last_output = Dense(1, name='last')(merge3)
# model = Model(inputs=[input1, input21, input31], outputs=last_output)

#2-5. 분기 1
dense41 = Dense(40, activation='relu', name='han41')(merge3)
dense42 = Dense(30, activation='relu', name='han42')(dense41)
dense43 = Dense(20, activation='relu', name='han43')(dense42)
dense44 = Dense(10, activation='relu', name='han44')(dense43)
output1 = Dense(1, name='han45')(dense44)

#2-6. 분기 2
dense51 = Dense(50, activation='relu', name='han51')(merge3)
dense52 = Dense(50, activation='relu', name='han52')(dense51)
dense53 = Dense(50, activation='relu', name='han53')(dense52)
dense54 = Dense(50, activation='relu', name='han54')(dense53)
output2 = Dense(1, name='han55')(dense54)

#2-7. model
model = Model(inputs=[input1, input21, input31], outputs=[output1, output2])

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
filepath = ''.join([path, 'keras69_3_', date, '-', filename])

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
    [x1_train, x2_train, x3_train], [y1_train, y2_train],
    epochs=3000,
    verbose=1,
    validation_split=0.25,
    callbacks=[es, mcp, rlr]
)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("loss : ", results)

x1_pred = np.array([range(100, 106), range(400, 406)]).T    # (6, 2)
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T   # (6, 3)
x3_pred = np.array([range(100, 106), range(400, 406), range(177, 183), range(133, 139)]).T # (6, 3)

y_pred = model.predict([x1_pred, x2_pred, x3_pred])
# print(type(y_pred)) # <class 'list'>
print("x1_pred와 x2_pred의 예측값 1 : ", y_pred[0])
print("x1_pred와 x2_pred의 예측값 2 : ", y_pred[1])

# Results
# Epoch 1618: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 2/2 [==============================] - 0s 52ms/step - loss: 15.7384 - han45_loss: 0.6775 - han55_loss: 15.0609 - val_loss: 6.4270 - val_han45_loss: 0.7098 - val_han55_loss: 5.7172 - lr: 0.0010
# Epoch 1618: early stopping
# 1/1 [==============================] - 0s 17ms/step - loss: 4.6387 - han45_loss: 0.8001 - han55_loss: 3.8385
# loss :  [4.638660430908203, 0.8001402616500854, 3.838520050048828]
# 1/1 [==============================] - 0s 122ms/step
# <class 'list'>
# x1_pred와 x2_pred의 예측값 1 :  [[3094.6826]
#  [3098.361 ]
#  [3101.954 ]
#  [3105.824 ]
#  [3109.9417]
#  [3113.0442]]
# x1_pred와 x2_pred의 예측값 2 :  [[13077.175]
#  [13090.718]
#  [13103.102]
#  [13112.627]
#  [13128.158]
#  [13139.592]]