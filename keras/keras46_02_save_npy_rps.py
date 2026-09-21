from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datagen = ImageDataGenerator(
    rescale=1./255
)

path = './_data/image/'
xy = datagen.flow_from_directory(
    path+'rps',
    target_size=(100, 100),
    batch_size=10000,
    class_mode='categorical',   # categorical : one-hot encoding까지 해줌. sparse : 0, 1, 2 형태로 저장
    color_mode='rgb',
    shuffle=True
)

x = xy[0][0]
y = xy[0][1]

print(x.shape)  # (2048, 100, 100, 3)
print(y.shape)  # (2048, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

np_path = './_data/npy_rps/'
np.save(np_path + 'keras46_02_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_02_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_02_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_02_y_test.npy', arr=y_test)

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측