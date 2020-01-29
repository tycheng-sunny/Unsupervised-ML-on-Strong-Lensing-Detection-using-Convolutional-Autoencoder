#====================================================================#
# Extract the representative features by a Convolutional AutoEncoder #
#====================================================================#

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #for GPU usage
#import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import callbacks, optimizers

from matplotlib import pyplot as plt
from astropy.io import fits
from skimage import exposure
#------------------------------------------------------------------------#
import time
Tstart = time.time() #Timer start

## MODEL PRE-SETTING ##
input_img = Input(shape=(101, 101, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(128, (8, 8), activation='relu', padding='same')(input_img) #(101, 101, 128)
x = MaxPooling2D((2, 2), padding='same')(x) #(51, 51, 128)
x = Conv2D(64, (7, 7), activation='relu', padding='same')(x) #(51, 51, 64)
x = MaxPooling2D((2, 2), padding='same')(x) #(26, 26, 64)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) #(26, 26, 32)
x = MaxPooling2D((2, 2), padding='same')(x) #(13, 13, 32)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) #(13, 13, 16)
x = MaxPooling2D((2, 2), padding='same')(x) #(7, 7, 16)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) #(7, 7, 8)
x = MaxPooling2D((2, 2), padding='same')(x) #(4, 4, 8)

x = Flatten()(x)
x = Dense(units=64, activation='relu')(x)
x = Dense(units=32, activation='relu')(x)
encoded = Dense(units=24, activation='relu', name='embedding')(x)
x = Dense(units=32, activation='relu')(encoded)
x = Dense(units=64, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
x = Reshape((4, 4, 8))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) #(4, 4, 8)
x = UpSampling2D((2, 2))(x) #(8, 8, 8)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) #(8, 8, 16)
x = UpSampling2D((2, 2))(x) #(16, 16, 16)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) #(16, 16, 32)
x = UpSampling2D((2, 2))(x) #(32, 32, 32)
x = Conv2D(64, (3, 3), activation='relu')(x) #(30, 30, 64)
x = UpSampling2D((2, 2))(x) #(60, 60, 64)
x = Conv2D(128, (7, 7), activation='relu')(x) #(54, 54, 128)
x = UpSampling2D((2, 2))(x) #(108, 108, 128)
decoded = Conv2D(1, (8, 8), activation='sigmoid')(x) #(101, 101, 1)

autoencoder = Model(input_img, decoded)
optimizer_adam = optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=optimizer_adam, loss='binary_crossentropy')
print(autoencoder.summary())
## END ##

#=================================================================#
## SETTING ##
#set-up
datapath = 'input_imgs/'
imagesize = (101, 101) #image size
Nepochs = 10 #number of epochs for CAE training
tosavemodel = True #if save the trained CAE model
plot_reconstruction = True #if plot the reconstruction comparison
savename = 'CAE_reconstruction' #setup if "tosavemodel=True" or "plot_reconstuction=True"

## DATE PREPARATION ##
listfn = os.listdir(datapath) #read the filename in the directory

# images to array for training
X_train = []
for i in listfn:
    imgname = datapath + i
    img = fits.open(imgname)
    img_data = img[0].data.astype(np.float32)
    img_data = exposure.rescale_intensity(img_data, in_range=(np.min(img_data), np.max(img_data)), out_range=(0, 1)) #normalise pixel values in each image
    X_train.append(img_data) #restore images to array
    img.close() #close the image file

X_train = np.stack(X_train)
X_train = X_train.reshape(len(X_train), imagesize[0], imagesize[1], 1) #change the shape to NHWC for CAE input
print(X_train.shape) #print information of training samples

Tprocess0 = time.time()
print('\n', '## DATE PREPARATION RUNTIME:', Tprocess0-Tstart) #Timer

## MAIN ##
#training
autoencoder.fit(X_train, X_train,
                epochs=Nepochs,
                shuffle=True)

Tprocess1 = time.time()
print('\n', '## CAE TRAINING RUNTIME:', Tprocess1-Tprocess0) #Timer

if tosavemodel:
    #restore the model
    autoencoder.save(savename + '.h5')

if plot_reconstruction:
    #plot the results
    decoded_imgs = autoencoder.predict(X_train)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(X_train[i].reshape(101, 101), cmap='gray_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i+1 + n)
        plt.imshow(decoded_imgs[i].reshape(101, 101), cmap='gray_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(savename + '.png')
    #plt.show()

## MAIN END ##
print('\n', '## CODE RUNTIME:', time.time()-Tstart) #Timer end
