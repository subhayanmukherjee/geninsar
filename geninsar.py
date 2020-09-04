# -*- coding: utf-8 -*-
"""
Author: Subhayan Mukherjee
Dated: 1st September, 2020
"""

import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"              # prevent training from hogging all GPUs

import keras
import mdn
import numpy as np
from numpy import expand_dims
from keras.layers import SeparableConv2D,Flatten,Dropout,Input
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import Model,load_model
from sklearn.feature_extraction.image import extract_patches_2d
import h5py
import math
import glob
from os.path import exists
import matplotlib.pyplot as plt
from data_utils import readFloatComplex,writeFloatComplex,writeFloat                                        # for reading and writing interferogram and coherence map files

folder_prefix = '/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/noisy_input/'                         # base path of all training interferogram folders

datafold   = ['folder1/','folder2/','folder3/','folder4/','folder5/','folder6/','folder7/']                 # folders containing training interferograms
df_cnt     = len(datafold)                                                                                  # number of folders in datafold
ifgsize    = [[1000,1000], [1500,1500], [2680,3820], [1000,1000], [4040,5060], [1000,1000], [1000,1000]]    # size of interferogram in each datafold folder
ifg_cnt    = [36,120,135,140,135,300,300]                                                                   # number of interferograms in each datafold folder
indices    = [True,False,False,False,False,False,True]                                                      # combine interferograms from more than one folder

checkpts_folder = './checkpts_grslreal_21x21'       # folder to save model checkpoints during training
logs_folder     = './logs_grslreal'                 # folder to save training logs to be used by tensorboard to monitor training if desired
outimage_folder = './outputs_real_21x21'            # folder to save prediction output images after each training epoch

N_MIXES     = 1             # number of mixture components in the model
OUTPUT_DIMS = 2             # number of values predicted by each mixture component: 1 real + 1 imaginary channel = 2

patch_size = 21
batch_size = 64
pat_per_img= batch_size     # adjust the constant 165 based on how many patches you want to use to train the model..

total_ifgs = np.sum(np.asarray(ifg_cnt) * np.asarray(indices))
total_pats = pat_per_img * total_ifgs
total_bats = total_pats // batch_size

hdf5_path = 'patches_dataset_' + str(patch_size) + 'x' + str(patch_size) + '.hdf5'    # path to where you want to save the training dataset hdf5 file

def lr_func(epoch, current_lr):
    drop = 0.5
    if epoch > 1 and math.log(epoch, 2).is_integer():
        current_lr = current_lr * drop
    print('\n    ##########  EPOCH: ' + str(epoch+1) + '; Learning rate: ' + str(current_lr) + '  ##########\n')
    return current_lr

def get_cnv(modelpt,image_size):
    model_cnv = Model(inputs=modelpt.input, outputs=modelpt.layers[9].output)
    
    N_FILTERS = 512
    kernel_size = 5
    model = keras.Sequential()
    model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//1 , kernel_size=kernel_size+0, activation='elu', padding='same', input_shape=(image_size,image_size,2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//2 , kernel_size=kernel_size+0, activation='elu', padding='same'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//4 , kernel_size=kernel_size+0, activation='elu', padding='same'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//8 , kernel_size=kernel_size+0, activation='elu', padding='same'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//16, kernel_size=kernel_size+0, activation='elu', padding='same'))
    model.set_weights(model_cnv.get_weights())
    
    return model

def get_fcm(modelpt):
    DL_input = Input(modelpt.layers[10].input_shape[1:])
    DL_model = DL_input
    for layer in model.layers[10:]:
        DL_model = layer(DL_model)
    return Model(inputs=DL_input, outputs=DL_model)

class Histories(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    model_cnv = get_cnv(self.model,1000)
    model_fcm = get_fcm(self.model)
    
    datafold = 'folder1/'
    ifgsize = [1000,1000]
    fname_ifg = '20130909_20130920.ifg'
    path_in_str = folder_prefix + datafold + fname_ifg      # perform prediction using this interferogram after each epoch of model training..
    
    ifg_image = readFloatComplex( path_in_str, ifgsize[0]*ifgsize[1] )
    ifg_image = np.reshape( ifg_image, (ifgsize[0],ifgsize[1]) )
    
    ifg_phase = np.angle(ifg_image)
    ifgreal   = np.cos(ifg_phase)
    ifgimag   = np.sin(ifg_phase)
    
    ifgreim   = np.expand_dims(np.concatenate((np.expand_dims(ifgreal,axis=-1), np.expand_dims(ifgimag,axis=-1)),axis=-1),axis=0)
    pred_parm = model_cnv.predict(ifgreim)
    pred_parm = np.squeeze(pred_parm)
    inpatches = np.reshape(pred_parm, (-1,pred_parm.shape[-1]))
    pred_parm = model_fcm.predict(inpatches,batch_size=4096)        # adjust batch size based on available GPU memory..
    
    pred_phas    = np.angle( pred_parm[:,0] + 1j*pred_parm[:,1] )
    pred_sigs    = np.sqrt(np.square(pred_parm[:,2]) + np.square(pred_parm[:,3]))
    pred_sigs[pred_sigs > 1.0] = 1.0
    pred_sigs[pred_sigs < 0.0] = 0.0
    pred_cohs    = np.sqrt(1 - pred_sigs**2.)
    
    plt.imsave(outimage_folder + '/proposed_phase_' + str(epoch+1) + '.png',np.reshape(pred_phas,(1000,1000)),vmin=-np.pi,vmax=np.pi,cmap='jet')
    plt.imsave(outimage_folder + '/proposed_variance_' + str(epoch+1) + '.png',np.reshape(pred_cohs,(1000,1000)),vmin=0.0,vmax=1.0,cmap='Greys_r')
    
    writeFloatComplex(outimage_folder + '/proposed_phase_' + str(epoch+1) + '.filt',np.cos(pred_phas)+1j*np.sin(pred_phas))
    writeFloat(outimage_folder + '/proposed_variance_' + str(epoch+1) + '.coh',pred_cohs)
    
    ### Generate more interferograms by sampling from the predicted Gaussian ###
    if (epoch+1)%15 == 0:      # sampling is time-consuming, so do occassionally
        for gdx in range(5):
            y_samples = np.squeeze( np.apply_along_axis(mdn.sample_from_output, 1, pred_parm, OUTPUT_DIMS, N_MIXES, sigma_temp=1.0))    # adjust sigma_temp between 0 and 1
            pred_phas = np.angle( y_samples[:,0] + 1j*y_samples[:,1] )
            plt.imsave(outimage_folder + '/proposed_phase_' + str(epoch+1) + '_' + str(gdx+1) + '.png',np.reshape(pred_phas,(1000,1000)),vmin=-np.pi,vmax=np.pi,cmap='jet')
    
    return

def extract_subpats(ifg_phase, pat_per_subimg):
    ifgreal   = np.cos(ifg_phase)
    ifgimag   = np.sin(ifg_phase)
    
    realpats  = extract_patches_2d(ifgreal, (patch_size, patch_size), pat_per_subimg, random_state=0)
    imagpats  = extract_patches_2d(ifgimag, (patch_size, patch_size), pat_per_subimg, random_state=0)
    
    targreal  = np.copy(realpats)
    targimag  = np.copy(imagpats)
    
    mid_pixel = patch_size // 2
    
    realpats[:,mid_pixel,mid_pixel] = 0     # avoid learning identity mapping
    imagpats[:,mid_pixel,mid_pixel] = 0     # avoid learning identity mapping
    
    realpats = np.expand_dims(realpats,axis=-1)
    imagpats = np.expand_dims(imagpats,axis=-1)
    reimpats = np.concatenate((realpats,imagpats),axis=-1)
    
    targreal = np.squeeze( targreal[:,mid_pixel,mid_pixel] )
    targimag = np.squeeze( targimag[:,mid_pixel,mid_pixel] )
    targpats = np.concatenate((np.expand_dims(targreal,axis=-1),np.expand_dims(targimag,axis=-1)),axis=-1)
    
    return reimpats, targpats

def build_dataset():
    # open a hdf5 file and create arrays
    if not exists(hdf5_path):
        print('Creating dataset...')
        hdf5_file = h5py.File(hdf5_path, mode='w')
        image_shape = (total_ifgs*pat_per_img, patch_size, patch_size, 2)
        label_shape = (total_ifgs*pat_per_img, 2)
        hdf5_file.create_dataset("train_img", image_shape, np.float32)
        hdf5_file.create_dataset("train_lab", label_shape, np.float32)
        
        img_idx = 0
        # extract patches from each trainig interferogram folder mentioned in datafold that is set to True in indices
        for idx in range(df_cnt):
            if not indices[idx]:
                continue
            sel_data = datafold[idx]
            print('Processing folder ' + sel_data)
            
            base_folder = folder_prefix + sel_data
            ifglist = glob.glob(base_folder + '*.ifg')
            examples_cnt = len(ifglist)
            for loop_idx in range(0, examples_cnt):
                path_in_str = ifglist[loop_idx]
                fname_parts = path_in_str.split("/")
                fname_ifg   = fname_parts[-1]
                
                ifg_image = readFloatComplex( path_in_str, ifgsize[idx][0]*ifgsize[idx][1] )
                ifg_image = np.reshape( ifg_image, (ifgsize[idx][0],ifgsize[idx][1]) )
                
                ifg_phase = np.angle(ifg_image)
                
                # Switch True to False below to turn off subsampling
                if True:    # Subsample Full-Resolution to Half-Resolution to reduce effect of (non-ideal) Point Spread Function (center pixel smudging out into neighboring pixels)
                    sub_reim_pats1, sub_targ_pats1 = extract_subpats( ifg_phase[1::2,1::2], pat_per_img // 4)     # subsampling FR to HR to fix PSF issue
                    sub_reim_pats2, sub_targ_pats2 = extract_subpats( ifg_phase[0::2,1::2], pat_per_img // 4)     # subsampling FR to HR to fix PSF issue
                    sub_reim_pats3, sub_targ_pats3 = extract_subpats( ifg_phase[1::2,0::2], pat_per_img // 4)     # subsampling FR to HR to fix PSF issue
                    sub_reim_pats4, sub_targ_pats4 = extract_subpats( ifg_phase[0::2,0::2], pat_per_img // 4)     # subsampling FR to HR to fix PSF issue
                    
                    reimpats  = np.concatenate((sub_reim_pats1,sub_reim_pats2,sub_reim_pats3,sub_reim_pats4), axis=0)
                    targpats  = np.concatenate((sub_targ_pats1,sub_targ_pats2,sub_targ_pats3,sub_targ_pats4), axis=0)
                else:
                    reimpats, targpats = extract_subpats( ifg_phase, pat_per_img )
                
                # save the patches (and labels) extracted from current training image
                hdf5_file["train_img"][img_idx*pat_per_img : (img_idx+1)*pat_per_img, ...] = reimpats
                hdf5_file["train_lab"][img_idx*pat_per_img : (img_idx+1)*pat_per_img, ...] = targpats
                
                img_idx += 1
        
        hdf5_file.close()

# The below function reads batches from the training dataset hdf5 file and passes them to the model.fit_generator() function
# This is to avoid having to read in the whole dataset into the main memory at once..
def generate_data(hdf5_file):
    while 1:
        rand_idx = np.random.permutation(total_bats-1)*batch_size
        loop_idx = 0
        while loop_idx < len(rand_idx):
            data = hdf5_file["train_img"][rand_idx[loop_idx]:(rand_idx[loop_idx]+batch_size)]
            labels = hdf5_file["train_lab"][rand_idx[loop_idx]:(rand_idx[loop_idx]+batch_size)]
            
            yield(data, labels)
            loop_idx += 1

build_dataset()        # just need to call once to build dataset, can comment out after that..

kernel_size = 5        # size of cnn filters to learn
N_FILTERS   = 512      # number of filters in the first convolutional layer

model = keras.Sequential()
model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//1 , kernel_size=kernel_size, activation='elu', padding='valid', input_shape=(patch_size,patch_size,2)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//2 , kernel_size=kernel_size, activation='elu', padding='valid'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//4 , kernel_size=kernel_size, activation='elu', padding='valid'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//8 , kernel_size=kernel_size, activation='elu', padding='valid'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.SeparableConv2D(filters=N_FILTERS//16, kernel_size=kernel_size, activation='elu', padding='valid'))
model.add(Flatten())
model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))

model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES), optimizer=keras.optimizers.Adam())
model.summary()

if not os.path.isdir(checkpts_folder):
    os.mkdir(checkpts_folder)
if not os.path.isdir(outimage_folder):
    os.mkdir(outimage_folder)

checkpointer = ModelCheckpoint(filepath=checkpts_folder + '/weights.{epoch:02d}.hdf5', verbose=1)	# , period=10)      # if you want to save model after every 10th epoch
tensorboard = TensorBoard(log_dir=logs_folder)
lr_schd = keras.callbacks.LearningRateScheduler(lr_func)
histories = Histories()

if False:       # if you want to load a previously saved model, set the if condition to True, and change the next two lines of code accoding to your requirements..
    model = load_model(checkpts_folder + '/weights.36.hdf5', custom_objects={'MDN': mdn.MDN, 'mdn_loss_func': mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)}, compile=False)
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES), optimizer=keras.optimizers.Adam(learning_rate=0.001/32))

with h5py.File(hdf5_path,'r') as hdf5_file:
    model.fit_generator(generator=generate_data(hdf5_file), initial_epoch=0, steps_per_epoch=total_bats, epochs=1000, max_queue_size=10, callbacks=[checkpointer,tensorboard,lr_schd,histories])
