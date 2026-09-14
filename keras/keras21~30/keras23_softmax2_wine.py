from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
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

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
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
# 
# Epoch 357/3000
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 0.9528 - loss: 0.1246 - val_acc: 0.9444 - val_loss: 0.1103
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 0.9722 - loss: 0.1414 
# loss :  0.14142635464668274
# acc :  0.97
# 걸린 시간 :  24.74 초
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step
# acc :  0.9722