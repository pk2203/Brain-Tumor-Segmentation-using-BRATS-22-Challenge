import os
import datetime
import numpy as np
import pandas as pd
import random
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from PIL import Image as im
import matplotlib.pyplot as plt
import cv2
import nibabel as nib


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=2, img_size=128, n_channels=3,
                 n_classes=4, shuffle=True):
        'Initialization'
        self.image_size = img_size
        self.dim = (img_size,img_size)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        volume_slices = 100
        volume_start = 22

        X = np.zeros((self.batch_size * volume_slices, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * volume_slices, 240, 240))
        Y = np.zeros((self.batch_size * volume_slices, *self.dim, 4))

        for c, i in enumerate(list_IDs_temp):
            case_path = os.path.join(r"/media/satya/My Passport/Personal_User/BITS/third year/Design Project/TASK1/Dataset/train", i)

            data_path = os.path.join(case_path, f'{i}_t1ce.nii.gz');
            ce = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_t2.nii.gz');
            t = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_flair.nii.gz');
            flair = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_seg.nii.gz');
            seg = nib.load(data_path).get_fdata()

            for j in range(volume_slices):
                X[j + volume_slices * c, :, :, 0] = cv2.resize(ce[:, :, j + volume_start],
                                                               (self.image_size, self.image_size));
                X[j + volume_slices * c, :, :, 1] = cv2.resize(t[:, :, j + volume_start],
                                                               (self.image_size, self.image_size));
                X[j + volume_slices * c, :, :, 2] = cv2.resize(flair[:, :, j + volume_start],
                                                               (self.image_size, self.image_size));

                y[j + volume_slices * c] = seg[:, :, j + volume_start];

        # Generate masks
        
        y[y == 4] = 3;
        mask = tf.one_hot(y, 4);
        X = preprocess_img(X)
        Y = tf.image.resize(mask, (self.image_size, self.image_size));

        return X, Y

def get_img_data(flair,ce,t):
    X = np.empty((100, 128, 128, 3))
    
    for j in range(100):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+22], (128,128))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+22], (128,128))
        X[j,:,:,2] = cv2.resize(t[:,:,j+22], (128,128))
    
    X = preprocess_img(X)
    return X

def img_normalize(image):
    image = tf.cast(image, tf.float32)
    image = image / np.max(image)
    return image

def preprocess_img(image):
    image = tf.image.random_flip_left_right(image)
    image = img_normalize(image)
    return image

def load_images(path,buffer_size,batch_size):
    t2_list = glob.glob(os.path.join(path, '*', '*_t2.nii.gz'))
    t1ce_list = glob.glob(os.path.join(path, '*', '*_t1ce.nii.gz'))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    t2_list = t2_list.map(preprocess_img, num_parallel_calls=AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)
    t1ce_list = t1ce_list.map(preprocess_img, num_parallel_calls=AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)

    return t2_list,t1ce_list

def get_steps_per_epoch(train_img_list, val_img_list, batch_size):
    steps_per_epoch = len(train_img_list)//batch_size
    val_steps_per_epoch = len(val_img_list)//batch_size

    if len(train_img_list) % batch_size != 0:
        steps_per_epoch += 1
    if len(val_img_list) % batch_size != 0:
        val_steps_per_epoch += 1

    return steps_per_epoch, val_steps_per_epoch

def get_images(path):
        images = os.listdir(path)

        for scan in images:
            f = os.path.join(path,scan)
            type = scan.split("\\")[-1].split('_')[-1]

            if type == 'flair.nii.gz' :
                img_flair = nib.load(f).get_fdata()

            if type == 't1ce.nii.gz':
                img_tice = nib.load(f).get_fdata()
            if type == 't1.nii.gz':
                img_t1 = nib.load(f).get_fdata()
            if type == 't2.nii.gz':
                img_t2 = nib.load(f).get_fdata()
            else:
                img_mask = nib.load(f).get_fdata()
                img_mask = img_mask.astype(np.uint8)
                img_mask[img_mask == 4] = 3

        n_slice = int(img_mask.shape[2]/2)

        plt.figure(figsize=(12, 8))

        plt.subplot(231)
        plt.imshow(img_flair[:, :, n_slice], cmap='gray')
        plt.title('Image flair')
        plt.subplot(232)
        plt.imshow(img_t1[:, :, n_slice], cmap='gray')
        plt.title('Image t1')
        plt.subplot(233)
        plt.imshow(img_tice[:, :, n_slice], cmap='gray')
        plt.title('Image t1ce')
        plt.subplot(234)
        plt.imshow(img_t2[:, :, n_slice], cmap='gray')
        plt.title('Image t2')
        plt.subplot(235)
        plt.imshow(img_mask[:, :, n_slice])
        plt.title('Mask')

        plt.savefig(
                os.path.join(r"D:\Personal_User\BITS\third year\Design Project\BraTS\matplot", 'TASK1_train' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'))



def get_xy_set(img_dir, split=0.2,train_size=1000):
    img = os.listdir(img_dir)
    indexes = np.random.choice(len(img),train_size)

    total_size = train_size
    valid_size = int(split * total_size)
    img_list = [img[i] for i in indexes]

    train, valid_ids = train_test_split(img_list, test_size=valid_size, random_state=42)

    train_ids, test_ids = train_test_split(train, test_size=valid_size, random_state=123)

    print(f"Train: Number of train ids = {len(train_ids)}\n")
    print(f"Valid: Number of valid ids = {len(valid_ids)}\n")
    print(f"Test: Number of test ids = {len(test_ids)}\n")

    return train_ids,valid_ids,test_ids
#
#target_cases = os.listdir(r"D:\Personal_User\BITS\third year\Design Project\TASK1\Dataset\train")
# img_delete(target_cases)  -- Got 620
#train_ids, valid_ids, test_ids= get_xy_set(r"/media/satya/My Passport/Personal_User/BITS/third year/Design Project/TASK1/Dataset/train")
#training_dataset = DataGenerator(train_ids, 2, 96)
#X,y = training_dataset[0]
#print(X.shape,y.shape)
# get_images(r"D:\Personal_User\BITS\third year\Design Project\TASK1\Dataset\train\BraTS2021_00014")
