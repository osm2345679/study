# 19-2 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.25,
    random_state=42
)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = min,
    patience = 70,
    restore_best_weights = False
)

hist = model.fit(x_train, y_train, epochs=2400, batch_size=256, verbose=1, validation_data=(x_val, y_val), callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,4))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='green', label='val_loss')
plt.legend(loc='upper right')
plt.title('Diabetes Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss / Val_loss')
plt.grid()
plt.show()

# Results
# 
# Epoch 424/2400
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step - loss: 3104.3398 - val_loss: 2736.3418
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 2920.9802 
# loss :  2920.980224609375