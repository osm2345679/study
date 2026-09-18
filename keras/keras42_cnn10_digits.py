#28-10 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
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

x_train = x_train.reshape(-1,8,8,1)
x_test = x_test.reshape(-1,8,8,1)

#2. 모델 구성
model = Sequential()
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

model.add(Conv2D(64, (2,1), input_shape=(8,8,1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,1), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())    # 
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(10, activation='softmax'))

#3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor = 'val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=8,
    verbose=1,
    callbacks=[es],
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

plt.figure(figsize=(9,6))
plt.plot(hist.history['acc'], c='orange', label='accuracy')
plt.plot(hist.history['val_acc'], c='red', label='val_accuracy')
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Acc / Val_acc / Loss / Val_loss')
plt.legend(loc='lower center')
plt.grid()
plt.show()

# Results
# Epoch 609/1000
# 135/135 [==============================] - 1s 5ms/step - loss: 0.9154 - acc: 0.6769 - val_loss: 0.7534 - val_acc: 0.7750
# 12/12 [==============================] - 0s 3ms/step - loss: 0.7627 - acc: 0.7889
# loss :  0.7627302408218384
# acc :  1
# 걸린 시간 :  528.1 초
# 12/12 [==============================] - 0s 1ms/step
# acc :  0.7888888888888889