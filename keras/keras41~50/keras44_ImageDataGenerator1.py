from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1./255,   # . 표시 : 형변환 표시 안해도 되는데 해주는 편이 형변환했음을 보기 쉬움
    
    #### 데이터 증폭
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.1, # 평형이동
    height_shift_range=0.1,
    rotation_range=5,    # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=1.2,
    shear_range=0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동. 찌부되는 거.
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, # 경로
    target_size=(100,100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale',  # 흑백
    shuffle=True
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10,
    class_mode='binary', # 이진분류
    color_mode='grayscale',  # 흑백
    shuffle=False
    # shuffle=True  # test에서는 불필요. 안 건드는 게 맞음.
)
# Found 120 images belonging to 2 classes.

# print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000001BEFB287FA0>
# print(xy_train.next())  # 첫번째 보여짐
# print(xy_train.next())  # 그 다음(두번째) 보여짐

# print(xy_train[0][0])   # 첫번째 배치의 x 데이터
# print(xy_train[0][1])   # 첫번째 배치의 y 데이터

# print(xy_train[0][0].shape) # (10, 100, 100, 1)
# print(xy_train[0][1].shape) # (10,)

# print(xy_train[16][0])  # 에러남. 총 160장, 배치 16개라서

# print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>