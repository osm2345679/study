# 50-1 카피
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


#################### 데이터 증폭 ####################
datagen = ImageDataGenerator(
    rescale=1./255,   # . 표시 : 형변환 표시 안해도 되는데 해주는 편이 형변환했음을 보기 쉬움
    
    #### 데이터 증폭
    horizontal_flip=True,   # 수평 뒤집기 (좌우반전)
    # vertical_flip=True, # 수직 뒤집기 (상하반전)
    width_shift_range=0.1, # 평형이동
    height_shift_range=0.1,
    rotation_range=15,    # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.1,
    shear_range=0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동. 찌부되는 거.
    fill_mode='nearest'
)

augment_size = 100

print(x_train.shape)    # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)

# aaa = np.tile(x_train[0], augment_size)
# print(aaa.shape)  # (28, 2800)

aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)    # (100, 28, 28, 1)
# 아직까진 단순 복붙. 사용 못함.

xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False
).next()

print(xy_data)
print(type(xy_data))    # <class 'tuple'>

# print(xy_data.shape)    # AttributeError: 'ImageDataGenerator' object has no attribute 'shape'
print(len(xy_data)) # 2

print(xy_data[0].shape) # (100, 28, 28, 1)

print(xy_data[1].shape) # (100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측