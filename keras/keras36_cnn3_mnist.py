# 36-2 카피
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000,)
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test)) # 255 0

###### 스케일링 1
# x_train = x_train/255.  # python 기초 - . 붙여서 실수형으로 변환 (. 붙이면 .0으로 판단하고 실수로 처리함)
# x_test = x_test/255   # python3부터는 정수 나눗셈하면 기본적으로 실수 나온다고 함
# # 이미지 데이터인 거 아니까 그냥 255로 나누기

# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0

###### 스케일링 2
x_train = (x_train-127.5)/127.5 # -1~1로 스케일링. 나눈 다음 그 값에서 1빼는 식으로 해도 됨
x_test = (x_test-127.5)/127.5

print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test)) # 1.0 -1.0