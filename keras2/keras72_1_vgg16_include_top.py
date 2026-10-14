'''
전이학습
Transfer Learning

Feature Extraction
    - 사전 학습된 모델을 고정된 특징 추출기로 활용
    ㄴ 가중치 가져다 쓴다는 뜻(전이학습 관련해서는)

Pre-trained Model
사전 학습 모델

사전학습 모델 기반의 파인 튜닝(Fine-tuning)
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from tensorflow.keras.applications import VGG16

#2. 모델

##### 디폴트 #####
model = VGG16(
    weights='imagenet',
    include_top=True,
    input_shape=(224, 224, 3)
)
model.summary()
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
#  flatten (Flatten)           (None, 25088)             0         
                                                                 
#  fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
#  fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
#  predictions (Dense)         (None, 1000)              4097000   
                                                                 
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

model = VGG16(
    # weights='imagenet',   # defalut가 imagenet이라 꺼도 됌.
    include_top=False,  # False로 해야 input_shape 수정 가능   # fc1, fc2, fc3 날라감. 최종 dense output 조정 가능
    input_shape=(100, 100, 3)
)
model.summary()
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_2 (InputLayer)        [(None, 100, 100, 3)]     0         
                                                                 
#  block1_conv1 (Conv2D)       (None, 100, 100, 64)      1792      
                                                                 
#  block1_conv2 (Conv2D)       (None, 100, 100, 64)      36928     
                                                                 
#  block1_pool (MaxPooling2D)  (None, 50, 50, 64)        0         
                                                                 
#  block2_conv1 (Conv2D)       (None, 50, 50, 128)       73856     
                                                                 
#  block2_conv2 (Conv2D)       (None, 50, 50, 128)       147584    
                                                                 
#  block2_pool (MaxPooling2D)  (None, 25, 25, 128)       0         
                                                                 
#  block3_conv1 (Conv2D)       (None, 25, 25, 256)       295168    
                                                                 
#  block3_conv2 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
#  block3_conv3 (Conv2D)       (None, 25, 25, 256)       590080    
                                                                 
#  block3_pool (MaxPooling2D)  (None, 12, 12, 256)       0         
                                                                 
#  block4_conv1 (Conv2D)       (None, 12, 12, 512)       1180160   
                                                                 
#  block4_conv2 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
#  block4_conv3 (Conv2D)       (None, 12, 12, 512)       2359808   
                                                                 
#  block4_pool (MaxPooling2D)  (None, 6, 6, 512)         0         
                                                                 
#  block5_conv1 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_conv2 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_conv3 (Conv2D)       (None, 6, 6, 512)         2359808   
                                                                 
#  block5_pool (MaxPooling2D)  (None, 3, 3, 512)         0         
                                                                 
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________

####### include_top = False #######
# 여기서 top은 모델의 아래를 의미함.
# 다차원의 큰 크기의 input이 conv 거치면서 줄어들다 작은 output으로 나오는 게 피라미드 모양으로 볼 때 output이 top임.
#1. FC 없어짐(적절하게 추가)
#2. input_shape 변경 가능