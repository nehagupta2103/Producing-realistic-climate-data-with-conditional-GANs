import numpy as np
import h5py as h5
import copy as cp
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from src.preparation import data_preproc as preproc

def build_generator(latent_dim, conditioning_dim, target_shape):
    input_noise = Input(shape=(latent_dim,))
    input_conditioning = Input(shape=(conditioning_dim,))
    
    merged_input = Concatenate(axis=1)([input_noise, input_conditioning])
    
    x = Dense(256 * 8 * 8, activation='relu')(merged_input)
    x = Reshape((8, 8, 256))(x)
    
    x = Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    generated_output = Conv2DTranspose(target_shape[2], kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh')(x)

    generator = Model(inputs=[input_noise, input_conditioning], outputs=generated_output, name='Generator')
    return generator

def build_discriminator(input_shape, conditioning_dim):
    input_data = Input(shape=input_shape)
    input_conditioning = Input(shape=(conditioning_dim,))
    
    x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(input_data)
    
    x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    
    merged_input = Concatenate(axis=1)([x, input_conditioning])
    
    x = Dense(1, activation='linear')(merged_input)
    
    discriminator = Model(inputs=[input_data, input_conditioning], outputs=x, name='Discriminator')
    return discriminator

class CGAN(object):
    def __init__(self, latent_dim, conditioning_dim, target_shape, batch_size, optimizerG=None, optimizerC=None):
        self.generator = build_generator(latent_dim, conditioning_dim, target_shape)
        self.discriminator = build_discriminator(target_shape, conditioning_dim)
        
        if optimizerG is None:
            optimizerG = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        if optimizerC is None:
            optimizerC = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizerG)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizerC, metrics=['accuracy'])
        
        self.latent_dim = latent_dim
        self.conditioning_dim = conditioning_dim
        self.target_shape = target_shape
        self.batch_size = batch_size
        
        self.generator.summary()
        self.discriminator.summary()

    def train(self, epochs, X_train, conditioning_labels):
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            real_images = X_train[idx]
            real_labels = conditioning_labels[idx]

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            generated_images = self.generator.predict([noise, real_labels])

            valid = np.ones((self.batch_size, 1))
            fake = np.zeros((self.batch_size, 1))

            d_loss_real = self.discriminator.train_on_batch([real_images, real_labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([generated_images, real_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            g_loss = self.generator_model.train_on_batch([noise, real_labels], valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

# Load the dataset
X_train, scaling = preproc.dataExtraction_puma(DB_path='./data/raw/data_plasim_3y_sc.h5', DB_name='dataset', im_shape=(64, 128, 81))

# Assuming you have labels for conditioning (replace it with your actual labels)
conditioning_labels = np.random.rand(X_train.shape[0], 10)

# Create and train the CGAN
cgan = CGAN(latent_dim=100, conditioning_dim=10, target_shape=(64, 128, 81), batch_size=64)
cgan.train(epochs=2000, X_train=X_train, conditioning_labels=conditioning_labels)
