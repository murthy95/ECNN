import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop

#creating a residual block
def residual_block(x_input):
    # x_input = Input(shape=(None, None, None, 64))

    conv1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(x_input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)

    # model = Model(inputs=a, outputs=b)
    # model.add()
    return Add(axis=-1)([bn2, Activation('relu')(x_input), x_input])

def build_ECNN():

    x_input = Input(shape=(None, None, None, 4))

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
    #attaching 13 residual blocks
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

max_iters = 40
batch_size = 32

sgd = SGD(lr=1e-2, decay=0.0005, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=1e-2, decay=0.0005, rho=0.9)
adam = Adam(lr=1e-2, decay = 0.0005, beta_1 = 0.9, beta_2 = 0.999)
