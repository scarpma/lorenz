#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_critic(fs, fm, init_sigma, init_mean):
    """ 
    fs = 20 dimensione filtro
    fm = 4 numero di filtri
    init_sigma = 0.2 varianza distribuzione normale per l'inizializzazione dei pesi del  modello
    init_mean = 0.0 media distribuzione normale per l'inizializzazione dei pesi del  modello
    """
    discriminator = Sequential()
    #2000x1
    discriminator.add(Conv1D(fm, fs, strides=4, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma), input_shape=(2000, 1)))
    discriminator.add(ReLU())
    #250x4
    discriminator.add(Conv1D(fm, fs, strides=4, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))    
    discriminator.add(ReLU())
    #50x4
    discriminator.add(Conv1D(fm, fs, strides=5, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma))) 
    discriminator.add(ReLU())
    #25x4
    discriminator.add(Conv1D(fm, fs, strides=5, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma))) 
    discriminator.add(ReLU())
    #5x4
    discriminator.add(Flatten())
    #20x1
    #discriminator.add(Dense(4*fm, activation='relu'))
    #64x1
    discriminator.add(Dense(1))
    #1x1
    #discriminator.summary()

    return discriminator
