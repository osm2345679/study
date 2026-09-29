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
print(arr)  # [[[ 87.  97.  96.] ... [222. 222. 222.]]]
print(arr.shape)    # (100, 100, 3)
print(type(arr))    # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0)   # (1, 100, 100, 3). 차원 증가
print(arr.shape)

arr /= 255

np_path = './_data/kaggle_cat_dog_npy/'

np.save(np_path + 'keras48_me_.npy', arr=arr)

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측