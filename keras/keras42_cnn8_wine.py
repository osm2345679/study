#23-2 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
import numpy as np
import datetime
import time

# acc = 0.95

#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)
print(x, y)

y = pd.get_dummies(y)
print(y)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=52,
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

x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)

#2. 모델 구성
model = Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))

model.add(Conv2D(64, (2,1), input_shape=(13,1,1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,1), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())    # 
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=8,
    verbose=1,
    callbacks=[es],
    validation_split=0.25
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", round(results[1], 2))

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_argmax)
print("acc : ", round(acc, 4))

# Results
# Epoch 264/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - acc: 0.8679 - loss: 0.3137 - val_acc: 0.8611 - val_loss: 0.3213
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - acc: 0.8611 - loss: 0.2860
# loss :  0.2860487103462219
# acc :  0.86
# 걸린 시간 :  29.57 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step
# acc :  0.8611