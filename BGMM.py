#===========================================================================================#
# Clustering the extracted features for each image at high-dimensional feature space by     #
# clustering algorithms, i.e. Bayesian Gaussian Mixture models in this task.                #
#===========================================================================================#
from __future__ import print_function
print(__doc__)

#----------------------------------------------------------------------
import os
import numpy as np
import itertools
import h5py
from matplotlib import pyplot as plt
from astropy.io import fits

from keras import backend as K
from keras.models import load_model

from sklearn.mixture import BayesianGaussianMixture as BGM
from skimage import exposure
import pickle
#----------------------------------------------------------------------
import time
Tstart = time.time() #Timer start
#=================================================================#
## SETTING ##
#set-up
datapath = 'input_imgs/'
cae_mfn = 'pretrained_CAE_model.h5'
imagesize = (101, 101) #image size
nfeature = 20 #number of extracted features in each image
nCluster = 20 #number of obtained clusters
tosavemodel = True #if save the trained BGMM model
savename = 'BGMM_model' #setup if "tosavemodel=True"
dirname = savename + '_group/' #directory for the output of brief examination

## DATE PREPARATION ##
listfn = os.listdir(datapath) #read the filename in the directory

# images to array for training
X_train, id_train = [], []
for i in listfn:
    imgname = datapath + i
    img = fits.open(imgname)
    img_data = img[0].data.astype(np.float32)
    img_data = exposure.rescale_intensity(img_data, in_range=(np.min(img_data), np.max(img_data)), out_range=(0, 1)) #normalise pixel values in each image
    X_train.append(img_data) #restore images to array
    id_train.append(imgname[13:-12])
    img.close() #close the image file

id_train = np.array(id_train)
X_train = np.stack(X_train)
X_train = X_train.reshape(len(X_train), imagesize[0], imagesize[1], 1) #change the shape to NHWC for CAE input
print(X_train.shape) #print information of training samples

Tprocess0 = time.time()
print('\n', '## DATE PREPARATION RUNTIME:', Tprocess0-Tstart) #Timer

## MAIN ##
#load CAE model
cae_model = load_model(cae_mfn)
#Retrieve the ecoder layer
Embedding_layer = K.function([cae_model.layers[0].input], [cae_model.layers[14].output])
input4bgmm = Embedding_layer([X_train[:]])
input4bgmm = np.array(input4bgmm)
input4bgmm = input4bgmm[0]
print(input4bgmm.shape)

#clustering
grouper = BGM(n_components=nCluster)
grouper.fit(input4bgmm)
if tosavemodel:
    #restore the model
    pickle.dump(grouper, open(savename, 'wb'))

Tprocess1 = time.time()
print('\n', '## CLUSTERING RUNTIME:', Tprocess1-Tprocess0) #Timer end

#brief examination
y_pred = grouper.predict(input4bgmm)
y_max = np.max(y_pred)
y_proba = grouper.predict_proba(input4bgmm) #probability of being a certain group

#group = [(number of group members): images, group label, probability for each group]
group = [ [] for _ in range(y_max+1)]
id_group = [ [] for _ in range(y_max+1)]
group_noise = [] #not in any group
for ix in range(len(y_pred)):
    for ig in range(len(group)):
        if y_pred[ix] == ig:
            tmp = [X_train[ix].reshape(imagesize[0], imagesize[1]), y_proba[ix]]
            group[ig].append(tmp)
            id_group[ig].append(id_train[ix])
        elif y_pred[ix] == -1:
            tmp = [X_train[ix].reshape(imagesize[0], imagesize[1]), y_proba[ix]]
            group_noise.append(tmp)
        else:
            continue

if not len(group_noise) == 0:
    group_noise = np.stack(group_noise)

for i in range(len(group)):
    if not len(group[i]) == 0:
        group[i] = np.stack(group[i])
group = np.array(group)

#plot random images in each group
if not os.path.isdir(dirname):
    os.mkdir(dirname)

for n in range(len(group)):
    #Output the examples in each group. Output 25 examples for each group unless the number of the group is less than 25, then output all of them.
    fig = plt.figure(figsize=(15,15))
    if len(id_group[n]) >= 25:
        ax1 = [None]* 25
        random_id = np.random.choice(id_group[n], 25, replace=False)
        for jshow in range(len(random_id)):
            loc = id_group[n].index(random_id[jshow])
            ax1[jshow] = fig.add_subplot(5,5,jshow+1)
            ax1[jshow].imshow(group[n][loc][0], cmap='gray_r')
            ax1[jshow].set_xticks([])
            ax1[jshow].set_xticklabels([])
            ax1[jshow].set_xlabel(str(random_id[jshow]))
            cur_axes = plt.gca()
            cur_axes.axes.get_yaxis().set_visible(False)
        plt.savefig(dirname + 'g' + str(n) + '.png')
    else:
        ax1 = [None]* len(id_group[n])
        for jshow in range(len(id_group[n])):
            loc = id_group[n].index(id_group[n][jshow])
            ax1[jshow] = fig.add_subplot(5,5,jshow+1)
            ax1[jshow].imshow(group[n][loc][0], cmap='gray_r')
            ax1[jshow].set_xticks([])
            ax1[jshow].set_xticklabels([])
            ax1[jshow].set_xlabel(str(id_group[n][jshow]))
            cur_axes = plt.gca()
            cur_axes.axes.get_yaxis().set_visible(False)
        plt.savefig(dirname + 'g' + str(n) + '.png')

if not len(group_noise) == 0:
    fig = plt.figure(figsize=(10,10))
    ax2 = [None]* 25
    for jshow in range(25):
        ax2[jshow] = fig.add_subplot(5,5,jshow+1)
        ax2[jshow].imshow(group_noise[jshow][0], cmap='gray_r')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
    plt.savefig(dirname + 'g_noise.png')

#time
Tprocess2 = time.time()
print('\n', '## EXAMINATION RUNTIME:', Tprocess2-Tprocess1) #Timer end

## END MAIN ##
print('\n', '## UML TRAINING RUNTIME:', time.time()-Tstart) #Timer end

