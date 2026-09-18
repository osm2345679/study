#34-10 카피
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,StandardScaler, MaxAbsScaler, RobustScaler
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import time

#1. 데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y = pd.get_dummies(y, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(256, input_dim=64))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))

# model.summary()

input1 = Input(shape=(64,))
dense1 = Dense(256)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(256, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(256, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(256, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(256, activation='relu')(drop4)
drop5 = Dropout(0.2)(dense5)
dense6 = Dense(256, activation='relu')(drop5)
drop6 = Dropout(0.2)(dense6)
dense7 = Dense(256, activation='relu')(drop6)
drop7 = Dropout(0.2)(dense7)
output1 = Dense(10, activation='softmax')(drop7)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일 , 훈련
path = './_save/keras31/'
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k31_10_", date, "-", filename])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor = 'val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)

start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=8,
    verbose=1,
    validation_split=0.25
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1]))

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_argmax)
print("acc : ", acc)

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['acc'], c='orange', label='accuracy')
# plt.plot(hist.history['val_acc'], c='red', label='val_accuracy')
# plt.plot(hist.history['loss'], c='green', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.xlabel('Epochs')
# plt.ylabel('Acc / Val_acc / Loss / Val_loss')
# plt.legend(loc='lower center')
# plt.grid()
# plt.show()

# Results
# 
# Epoch 229/3000
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 1.0000 - loss: 2.6203e-06 - val_acc: 0.9972 - val_loss: 0.0139
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9722 - loss: 0.1789 
# loss :  0.1788996160030365
# acc :  1
# 걸린 시간 :  21.82 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# acc :  0.9722222222222222
#
# MinMaxScaler
# Epoch 187/3000
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 1.0000 - loss: 1.3430e-05 - val_acc: 0.9972 - val_loss: 0.0029
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 836us/step - acc: 0.9750 - loss: 0.1637
# loss :  0.16372902691364288
# acc :  1
# 걸린 시간 :  17.37 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# acc :  0.975
#
# StandardScaler + Fine-tuning
# Epoch 325/190000
# 68/68 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 1.0000 - loss: 6.6412e-10 - val_acc: 0.9917 - val_loss: 0.0692
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9778 - loss: 0.1304 
# loss :  0.13043555617332458
# acc :  1
# 걸린 시간 :  41.78 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc :  0.9777777777777777
#
# MaxAbsScaler
# Epoch 121/190000
# 135/135 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 1.0000 - loss: 0.0000e+00 - val_acc: 0.9889 - val_loss: 0.2497
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9833 - loss: 0.0661  
# loss :  0.0661407932639122
# acc :  1
# 걸린 시간 :  32.03 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc :  0.9833333333333333
#
# RobustScaler
# Epoch 121/1000
# 135/135 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 1.0000 - loss: 2.2137e-09 - val_acc: 0.9861 - val_loss: 0.1417
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9694 - loss: 0.1653  
# loss :  0.16527247428894043
# acc :  1
# 걸린 시간 :  34.65 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc :  0.9694444444444444
#
# CPU vs GPU
# Epoch 100/100
# 135/135 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9916 - loss: 0.0411 - val_acc: 0.9833 - val_loss: 0.1060
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.9667 - loss: 0.1919     
# loss :  0.19189965724945068
# acc :  1
# 걸린 시간 :  28.3 초
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc :  0.9666666666666667
# Epoch 100/100
# 135/135 [==============================] - 0s 2ms/step - loss: 0.0607 - acc: 0.9935 - val_loss: 0.3055 - val_acc: 0.9778
# 12/12 [==============================] - 0s 1ms/step - loss: 0.2012 - acc: 0.9722
# loss :  0.20121333003044128
# acc :  1
# 걸린 시간 :  32.93 초
# 12/12 [==============================] - 0s 259us/step
# acc :  0.9722222222222222