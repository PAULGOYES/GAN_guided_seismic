from tensorflow.keras.layers import Dense, ReLU, Input, Activation, BatchNormalization, MaxPooling2D, Cropping2D, UpSampling2D, Concatenate,Flatten, Conv2D, Conv2DTranspose, LeakyReLU,PReLU, add, ReLU, concatenate
from tensorflow.keras.layers import Flatten, Reshape, Add, GroupNormalization


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import keras

#visualización de imagenes
import matplotlib.pyplot as plt
import matplotlib
import numpy as np




def resblock(x, kernelsize, filters):
    fx = Conv2D(filters, kernelsize, padding='same')(x)
    fx = BatchNormalization(momentum = 0.5)(fx)
    fx = LeakyReLU()(fx)
    fx = Conv2D(filters, kernelsize, padding='same')(fx)
    fx = BatchNormalization(momentum = 0.5)(fx)
    out = Add()([x,fx])
    out = LeakyReLU()(out)
    return out
    
    
def ResNet():
    input_img = Input(shape=(28, 28, 64))
    resblock1 = resblock(input_img, (5,5),64)
    resblock2 = resblock(resblock1, (5,5),64)
    resblock3 = resblock(resblock2, (5,5),64)
    conv = Conv2D(1, (5,5), padding='same', activation='sigmoid')(resblock3)
    return Model(input_img, conv)
    
def Unet():
  input_img = Input(shape=(28, 28, 1))

  Econv1 = Conv2D(64,(5, 5),strides=(2, 2),padding='same', use_bias=False, kernel_regularizer='l1')(input_img)
  Econv1 = BatchNormalization(momentum = 0.5)(Econv1)
  Econv1 = LeakyReLU()(Econv1)

  Econv2 = Conv2D(128,(5, 5),strides=(2, 2),padding='same',use_bias=False,kernel_regularizer='l1')(Econv1)
  Econv2 = BatchNormalization(momentum = 0.5)(Econv2)
  Econv2 = LeakyReLU()(Econv2)


  Econv3 = Conv2D(256,(5,5),padding='same',use_bias=False,kernel_regularizer='l1')(Econv2)
  Econv3 = BatchNormalization(momentum = 0.5)(Econv3)
  Econv3 = LeakyReLU()(Econv3)

  ann1 = keras.layers.Flatten()(Econv3)
  encoded = Dense(10)(ann1)
  encoded = BatchNormalization(momentum = 0.5)(encoded)
  encoded = LeakyReLU()(encoded)

  # here begins decoder

  ann5 = Dense(7*7*256)(encoded)
  ann6 = Reshape((7,7,256))(ann5)
  ann6 = concatenate([ann6, Econv3], axis=3) #Skip connection
  ann6 = BatchNormalization(momentum = 0.5)(ann6)
  ann6 = LeakyReLU()(ann6)

  Dconv1 = Conv2DTranspose(128,(5, 5),strides=(1, 1),padding='same',use_bias=False,kernel_regularizer='l1')(ann6)
  Dconv1 = concatenate([Dconv1, Econv2], axis=3) #Skip connection
  Dconv1 = BatchNormalization(momentum = 0.5)(Dconv1)
  Dconv1 = LeakyReLU()(Dconv1)


  Dconv2 = Conv2DTranspose(64,(5, 5),strides=(2, 2),padding='same',use_bias=False,kernel_regularizer='l1')(Dconv1)
  Dconv2 = concatenate([Dconv2, Econv1]) #Skip connection
  Dconv2 = BatchNormalization(momentum = 0.5)(Dconv2)
  Dconv2 = LeakyReLU()(Dconv2)

  decoded = Conv2DTranspose(63, (5, 5),strides=(2, 2),padding='same',use_bias=False,kernel_regularizer='l1',name='outUnet')(Dconv2)
  decoded = concatenate([input_img, decoded])
  decoded = BatchNormalization(momentum = 0.5)(decoded)
  decoded = LeakyReLU()(decoded)

  return Model(input_img, decoded)



