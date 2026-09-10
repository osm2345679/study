#23-4 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

y = pd.get_dummies(y, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=64))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor = 'val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)
start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=32,
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