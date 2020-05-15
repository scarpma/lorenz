#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow
import tensorflow.keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Conv2DTranspose, Flatten, Dropout, ReLU, Input, MaxPooling1D, Reshape
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K

import numpy as np


# LOAD DATABASES, CONCATENATE AND MIX THEM

# In[2]:


def load_data():
    rv = [54.]
    nr = len(rv) 
    
    def load_and_shuffle_dbs(rv):
        paths = []
        for r in rv:
            paths.append(f"/scratch/scarpolini/databases/db_lorenz_{r:.1f}.npy")
        
        n_traj = 50000
        db = np.ndarray(shape=(nr*n_traj,2000,1))
        labels = []
        for path,r,i in zip(paths,rv,range(nr)):
            db1 = np.load(path)
            for j in range(n_traj):
                db[i*n_traj + j,:,0] = db1[j,0,:]
                labels.append(r)
        
        labels = np.array(labels)
        
        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]
        
        db, labels = unison_shuffled_copies(db, labels)
        return db, labels
    

    db, labels = load_and_shuffle_dbs(rv)
    
    
    
    
    
    validation_split = 0.0
    
    sig_len = len(db[0,:,0])
    print(f"siglen: {sig_len}")
    channels = 1 #len(db[0,0,:])
    print(f"channels: {channels}")
    n_traj = len(db[:,0,0])
    print((f"n_traj: {n_traj}"))
    # numero della prima traiettoria usata come validation
    first_validation = n_traj #   round((1. - validation_split)*n_traj)
    #  print(f"first_validation: {first_validation}")
    db_train = np.ndarray(shape=(first_validation,sig_len,channels))
    #db_test = np.ndarray(shape=(n_traj-first_validation,sig_len,channels))
    db_train = db[:first_validation,:,0:channels]
    #db_test = db[first_validation:,:,0:channels]
    del db
    
    print(db_train.shape)
    #M = np.max(np.append(db_train, db_test))
    #m = np.min(np.append(db_train, db_test))
    M = np.max(db_train)
    m = np.min(db_train)
    print(M,m)
    db_train = (db_train - m)/(M - m)
    #db_test = (db_test - m)/(M - m)
    M = np.max(db_train)
    m = np.min(db_train)
    print(M,m)
    
    return db_train


# In[3]:


class WGAN():
    def __init__(self):
        self.sig_len = 2000
        self.channels = 1
        self.noise_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.1
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.gen = self.build_generator()

        self.critic.trainable = False
        gan_input = Input(shape=(self.noise_dim,))
        fake_traj = self.gen(gan_input)
        gan_output = self.critic(fake_traj)
        #K.clear_session()
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        self.critic.trainable = True

        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        fs = (20,1) # dimensione filtro
        fm = 4 # numero filtri
        init_sigma = 0.2 # varianza distribuzione parametri iniziali dei kernel
        mean_init = 0.03
        alpha = 0.3
        #K.clear_session()
        generator = Sequential()
        # Starting size
        d = 5
        #4x1
        generator.add(Dense(d*fm, kernel_initializer=RandomNormal(mean_init, init_sigma), input_dim=(self.noise_dim)))
        generator.add(ReLU(negative_slope=alpha))
        #20x1
        generator.add(Reshape((d, 1, fm)))
        #5x4
        generator.add(Conv2DTranspose(fm, fs, strides=(5,1), padding='same', kernel_initializer=RandomNormal(mean_init, init_sigma)))
        generator.add(ReLU(negative_slope=alpha))
        #50x4
        generator.add(Conv2DTranspose(fm, fs, strides=(5,1), padding='same', kernel_initializer=RandomNormal(mean_init, init_sigma)))
        generator.add(ReLU(negative_slope=alpha))
        #250x4
        generator.add(Conv2DTranspose(fm, fs, strides=(4,1), padding='same', kernel_initializer=RandomNormal(mean_init, init_sigma)))
        generator.add(ReLU(negative_slope=alpha))
        #2000x1
        generator.add(Conv2DTranspose(1, fs, strides=(4,1), padding='same', kernel_initializer=RandomNormal(mean_init, init_sigma)))
        generator.add(Reshape((2000, 1)))
        generator.summary()
        return generator

    
    def build_critic(self):
        fs = 20 # dimensione filtro
        fm = 4 # numero filtri
        sigma_init = 0.2 # varianza distribuzione parametri iniziali dei kernel
        mean_init = 0.0
        #K.clear_session()
        discriminator = Sequential()
        #2000x1
        discriminator.add(Conv1D(fm, fs, strides=4, padding='same', kernel_initializer=RandomNormal(mean_init, sigma_init), input_shape=(self.sig_len, self.channels)))
        discriminator.add(ReLU())
        #250x4
        discriminator.add(Conv1D(fm, fs, strides=4, padding='same', kernel_initializer=RandomNormal(mean_init, sigma_init)))    
        discriminator.add(ReLU())
        #50x4
        discriminator.add(Conv1D(fm, fs, strides=5, padding='same', kernel_initializer=RandomNormal(mean_init, sigma_init))) 
        discriminator.add(ReLU())
        #25x4
        discriminator.add(Conv1D(fm, fs, strides=5, padding='same', kernel_initializer=RandomNormal(mean_init, sigma_init))) 
        discriminator.add(ReLU())
        #5x4
        discriminator.add(Flatten())
        #20x1
        #discriminator.add(Dense(4*fm, activation='relu'))
        #64x1
        discriminator.add(Dense(1))
        #1x1
        discriminator.summary()
        return discriminator


    def train(self, epochs, batch_size=250):
        abc = 0
        db_train = load_data()
        
        abc += 1
        print(abc)
        
        static_noise = np.random.normal(0, 1, size=(1, self.noise_dim))
        
        abc += 1
        print(abc)
                
        mini_batch_size = batch_size * self.n_critic
                
        abc += 1
        print(abc)
        
        steps_per_epoch = len(db_train[:,0,0]) // batch_size
                
        abc += 1
        print(abc)
        
        valid = -np.ones(batch_size)
                
        abc += 1
        print(abc)
        
        fake  =  np.ones(batch_size)
                
        abc += 1
        print(abc)
        
        disc_labels = np.concatenate((valid, fake))
        for epoch in range(epochs):
            for ii in range(len(db_train[:,0,0]) // mini_batch_size):
                real_trajs = db_train[np.arange(ii*mini_batch_size, (ii+1)*mini_batch_size)]
                        
                abc = 'real trajs'
                print(abc)
        

                for jj in range(self.n_critic):
                    noise = np.random.normal(0, 1, size=(batch_size, self.noise_dim))
                    abc = 'noise'
                    print(abc)
                    fake_trajs = self.gen.predict(noise)
                    abc = 'fake_traj'
                    print(abc)
                    trajs = np.concatenate((real_trajs[jj*batch_size:(jj+1)*batch_size], fake_trajs))
                        
                    abc = 'train critic'
                    print(abc)
        
                    self.critic.train_on_batch(trajs, disc_labels)
                    # Clip critic weights
                    for l in self.critic.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)
                                    
            abc = 'train gen'
            print(abc)
        
    
            self.gan.train_on_batch(noise, valid) # training del generatore (per imbrogliare il discrim.)
                                    
            abc = 'gen trained'
            print(abc)        

            print(f'{ii}', end='r')
        print(epoch)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#            # Load the dataset
#            (X_train, _), (_, _) = mnist.load_data()
#    
#            # Rescale -1 to 1
#            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#            X_train = np.expand_dims(X_train, axis=3)
#    
#            # Adversarial ground truths
#            valid = -np.ones((batch_size, 1))
#            fake = np.ones((batch_size, 1))
#    
#            for epoch in range(epochs):
#    
#                for _ in range(self.n_critic):
#    
#                    # ---------------------
#                    #  Train Discriminator
#                    # ---------------------
#    
#                    # Select a random batch of images
#                    idx = np.random.randint(0, X_train.shape[0], batch_size)
#                    imgs = X_train[idx]
#                    
#                    # Sample noise as generator input
#                    noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
#    
#                    # Generate a batch of new images
#                    gen_imgs = self.generator.predict(noise)
#    
#                    # Train the critic
#                    d_loss_real = self.critic.train_on_batch(imgs, valid)
#                    d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
#                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
#    
#                    # Clip critic weights
#                    for l in self.critic.layers:
#                        weights = l.get_weights()
#                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
#                        l.set_weights(weights)
#    
#    
#                # ---------------------
#                #  Train Generator
#                # ---------------------
#    
#                g_loss = self.combined.train_on_batch(noise, valid)
#    
#                # Plot the progress
#                print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
#    
#                # If at save interval => save generated image samples
#                if epoch % sample_interval == 0:
#                    self.sample_images(epoch)

#    def sample_images(self, epoch):
#        r, c = 5, 5
#        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
#        gen_imgs = self.generator.predict(noise)
#
#        # Rescale images 0 - 1
#        gen_imgs = 0.5 * gen_imgs + 0.5
#
#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
#                axs[i,j].axis('off')
#                cnt += 1
#        fig.savefig("images/mnist_%d.png" % epoch)
#        plt.close()


#if __name__ == '__main__':
wgan = WGAN()

wgan.train(epochs=1000)
