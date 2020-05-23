#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_generator(fs, fm, init_sigma, init_mean, alpha, noise_dim):
    """ 
    fs = (20,1) dimensione filtro
    fm = 4 numero di filtri
    init_sigma = 0.2 varianza distribuzione normale per l'inizializzazione dei pesi del  modello
    init_mean = 0.03 media distribuzione normale per l'inizializzazione dei pesi del  modello
    alpha = 0.3 pendenza parte negativa del leaky relu
    """









    generator = Sequential()
    # Starting size
    generator.add(Dense(800, kernel_initializer=RandomNormal(init_mean, init_sigma), input_dim=noise_dim))
    generator.add(ReLU(negative_slope=alpha))
    #20x1
    generator.add(Reshape((25, 1, 32)))
    #5x4
    generator.add(Conv2DTranspose(32, fs, strides=(5,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU(negative_slope=alpha))
    #50x4
    generator.add(Conv2DTranspose(16, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU(negative_slope=alpha))
    generator.add(Conv2DTranspose(8, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU(negative_slope=alpha))
    #50x4
    generator.add(Conv2DTranspose(4, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU(negative_slope=alpha))
    generator.add(Conv2DTranspose(2, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(ReLU(negative_slope=alpha))
    generator.add(Conv2D(1, fs, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))

    generator.add(Reshape((2000, 1)))
    generator.summary()











    #generator = Sequential()
    ## Starting size
    #d = 5
    ##4x1
    #generator.add(Dense(d*fm, kernel_initializer=RandomNormal(init_mean, init_sigma), input_dim=(noise_dim)))
    #generator.add(ReLU(negative_slope=alpha))
    ##20x1
    #generator.add(Reshape((d, 1, fm)))
    ##5x4
    #generator.add(Conv2DTranspose(fm, fs, strides=(5,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(ReLU(negative_slope=alpha))
    ##50x4
    #generator.add(Conv2DTranspose(fm, fs, strides=(5,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(ReLU(negative_slope=alpha))
    ##250x4
    #generator.add(Conv2DTranspose(fm, fs, strides=(4,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(ReLU(negative_slope=alpha))
    ##2000x1
    #generator.add(Conv2DTranspose(1, fs, strides=(4,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #generator.add(Reshape((2000, 1)))
    #
    #generator.summary()
    return generator



if __name__ == '__main__':
    fs = (20,1)
    fm = 4
    init_sigma = 0.02
    init_mean = 0.01
    alpha = 0.3
    noise_dim = 100
    gen = build_generator(fs, fm, init_sigma, init_mean, alpha, noise_dim)
