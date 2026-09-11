import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.7,
    test_size=0.3,  # 디폴트는 0.25. train_size와 합쳐서 1 이하인 건 문제 없지만 1 초과는 에러남.
    shuffle=True,   # 디폴트는 True
    random_state=42 # 지정 안하면 코드 실행될 때마다 train, test 리스트가 바뀜.
)

print(x_train, x_test, y_train, y_test) # [ 1  8  3 10  5  4  7] [9 2 6] [ 1  8  3 10  5  4  7] [9 2 6]

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=2)

print("=======================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

result = model.predict(np.array([11]))
print("[11]의 예측값 : ", result)

results = model.predict(x)
print("x의 예측값 : ", results)


# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)   # 데이터 점 찍기
plt.plot(results, color='red')
plt.show()


# Result
# 
# Epoch 1000/1000
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 3.1670e-13 
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 7.5791e-14
# loss :  7.579122740649855e-14
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
# [11]의 예측값 :  [[10.999999]]