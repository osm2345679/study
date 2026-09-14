from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. 내부 연산(Matrix Multiplication 등)에 사용할 스레드 수
tf.config.threading.set_intra_op_parallelism_threads(16)

# 2. 독립적인 연산들을 병렬로 처리할 스레드 수
tf.config.threading.set_inter_op_parallelism_threads(16)

#1. 데이터
path = './_data/kaggle_santander/'
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.info())
print(test_csv.info())

x = train_csv.drop('target', axis=1)
y = train_csv['target']

y_dummies = pd.get_dummies(y, dtype=int)
print(y_dummies)
print(y_dummies.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y_dummies,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
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
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_pred = model.predict(x_test)
y_argmax = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_argmax)
print("acc_score : ", acc_score)

plt.figure(figsize=(9,6))
plt.plot(hist.history['acc'], c='red', label='acc')
plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
plt.plot(hist.history['loss'], c='green', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Acc/Val_acc/Loss/Val_loss')
plt.legend(loc='upper right')
plt.title('Kaggle Santander Results')
plt.grid()
plt.show()

submit = model.predict(test_csv)
submit = np.round(submit)   # 이진분류이므로 라운드 처리
submission_csv['target'] = submit
submission_csv.to_csv(path + "submit/" + "0910_1040.csv")

# Results
#
# 3750/3750 ━━━━━━━━━━━━━━━━━━━━ 17s 4ms/step - acc: 0.9140 - loss: 0.2323 - val_acc: 0.9102 - val_loss: 0.2450
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - acc: 0.9106 - loss: 0.2417
# loss :  [0.24169084429740906, 0.9106249809265137]
# 걸린 시간 :  617.99 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step  
# acc_score :  0.910625
# 6250/6250 ━━━━━━━━━━━━━━━━━━━━ 8s 1ms/step