# 16-2 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.3,
    random_state=42
)

print(x_train, x_val, x_test, y_train, y_val, y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4, verbose=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

result = model.predict(x_test)
print("result : ", result)

# Results
# 
# Epoch 100/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step - loss: 0.0624 - val_loss: 0.0703
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 0.1728
# loss :  0.1727961301803589
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
# result :  [[ 0.46437994]
#  [ 1.5291487 ]
#  [ 5.7882233 ]
#  [15.3711405 ]]