# 48 카피
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
path = 'c:/study/_data/image/'
# path = './_data/image/'

img = load_img(path + 'id.jpg', target_size=(100,100))

# print(img)  # <PIL.Image.Image image mode=RGB size=100x100 at 0x1C407291660>
# print(type(img))    # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)  # [[[ 87.  97.  96.] ... [222. 222. 222.]]]
# print(arr.shape)    # (100, 100, 3)
# print(type(arr))    # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0)   # 차원 증가
# print(arr.shape)

# arr /= 255

# np_path = './_data/kaggle_cat_dog_npy/'

# np.save(np_path + 'keras48_me_.npy', arr=arr)

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

it = datagen.flow(arr, batch_size=1)
print(it)   # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001DC25F87F70>
# print(it.next())    # 파이썬 3.10까지
print(next(it))       # 파이썬 3.11 이후

print(next(it).shape)   # (1, 100, 100, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(100, 100, 3)

    ax[i].imshow(batch)
    # ax[i].show('off')
plt.show()

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측