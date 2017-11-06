import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, Input, Add
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import glob
from classes import DataGenerator

#creating a residual block
def residual_block(x_input):
    conv1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(x_input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)

    return Add()([bn2, Activation('relu')(x_input), x_input])

def build_ECNN():

    x_input = Input(shape=(None, None, 4))

    conv1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(x_input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)


    conv2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    conv3 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act2)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)

    activation = act3
    # attaching 13 residual blocks
    for i in range(13):
        concat = residual_block(activation)
        activation = concat

    conv_1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(activation)
    bn_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(bn_1)

    conv_2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act_1)
    bn_2 = BatchNormalization()(conv_2)
    act_2 = Activation('relu')(bn_2)

    conv_3 = Conv2D(1, kernel_size=(1,1), strides=(1, 1), padding='same')(act_2)

    model = Model(inputs=x_input, outputs=conv_3)

    return model

def load_data(file_path):
    with file_path.open() as file:
        data = file.read_lines()
    return data

def read_paths(file_path):
    with open(file_path, 'r') as data:
        return data.readlines()

#model summary
model = build_ECNN()
print model.summary()

#model parameters
max_iters = 40
batch_size = 32
sgd = SGD(lr=1e-2, decay=0.0005, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=1e-2, decay=0.0005, rho=0.9)
adam = Adam(lr=1e-2, decay = 0.0005, beta_1 = 0.9, beta_2 = 0.999)

#read paths of train, val and test images
train_ids = read_paths('Data/train_imgs.txt')
val_ids = read_paths('Data/val_imgs.txt')
test_ids = read_paths('Data/test_imgs.txt')
# os.chdir('Data/')
model.compile(optimizer=sgd, loss='mse')

params = {'dim_x': 224,
          'dim_y': 224,
          'dim_z': 4,
          'batch_size': 32,
          'shuffle': True}

train_generator = DataGenerator(**params).generate(train_ids)
val_generator = DataGenerator(**params).generate(val_ids)

model.fit_generator(generator = train_generator,
                    steps_per_epoch = len(train_ids)/batch_size,
                    validation_data = val_generator,
                    validation_steps = len(val_ids)/batch_size)
