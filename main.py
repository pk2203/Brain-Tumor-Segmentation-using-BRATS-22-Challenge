import datetime
import os
import glob
import random
import nibabel as nib
import tarfile
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import cv2
# import splitfolders

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pydicom as pyd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



class task1_utils:
    def __init__(self, size=128, normalize=False):
        self.training_data = r"D:\Personal_User\BITS\third year\Design Project\archive\BraTS2021_Training_Data.tar"
        self.root = r"D:\Personal_User\BITS\third year\Design Project\TASK1\Dataset\train"
        self.combinedimgs = r"D:\Personal_User\BITS\third year\Design Project\archive\Dataset\combined_imgs"
        self.validation = r"D:\Personal_User\BITS\third year\Design Project\RSNA_ASNR_MICCAI_BraTS2021_ValidationData\images_npy"
        self.output_dir = r"D:\Personal_User\BITS\third year\Design Project\TASK1\Dataset\output_dir"
        self.matplotimg = r"D:\Personal_User\BITS\third year\Design Project\BraTS\matplot"
        self.image_size = size
        self.normalize = normalize
        self.scaler = MinMaxScaler()

    def extract_task1_files(self):
        tar = tarfile.open(self.training_data)
        tar.extractall(self.root)
        tar.close()

    def read_file(self, target_case, show_imgs = False):
        """
        :param target_cases: Iterates through the nib extension files to access images under the different modalities
        :param seg_case: Gets the absolute path of the segmentation of the particular target case
        :return: Returns a dict for each scan type in (scans:,combined_channel:,segmentation:)
        """
        case_path = os.path.join(self.root, target_case)
        data_path = os.path.join(case_path, f'{target_case}_flair.nii.gz');
        flair = nib.load(data_path).get_fdata()
        flair = self.scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)
        flair = np.rot90(flair)

        data_path = os.path.join(case_path, f'{target_case}_t1ce.nii.gz');
        ce = nib.load(data_path).get_fdata()
        ce = self.scaler.fit_transform(ce.reshape(-1, ce.shape[-1])).reshape(ce.shape)
        ce = np.rot90(ce)

        data_path = os.path.join(case_path, f'{target_case}_t2.nii.gz');
        t = nib.load(data_path).get_fdata()
        t = self.scaler.fit_transform(t.reshape(-1, t.shape[-1])).reshape(t.shape)
        t = np.rot90(t)

        data_path = os.path.join(case_path, f'{target_case}_t1.nii.gz');
        t1 = nib.load(data_path).get_fdata()
        t1 = self.scaler.fit_transform(t1.reshape(-1, t1.shape[-1])).reshape(t1.shape)
        t1 = np.rot90(t1)

        data_path = os.path.join(case_path, f'{target_case}_seg.nii.gz');
        seg = nib.load(data_path).get_fdata()
        seg = np.clip(seg,0,255).astype('uint8')
        seg = np.rot90(seg)
        seg[seg == 4] = 3
        mask = to_categorical(seg,num_classes=4)
      #  seg = self.preprocess_mask(seg)
        val, count = np.unique(mask, return_counts=True)
        print(count,val)
        percen = 1 - (count[0] / count.sum())

        if show_imgs:
            self.get_images(flair,t1,ce,t,seg,target_case)

        return flair,t1,ce,t,mask,percen

    def get_images(self, img_flair, img_t1,img_tice,img_t2,img_mask,case):

        n_slice = int(img_mask.shape[2]/2)
        print(n_slice)
        case = case.split('_')[1]
        plt.figure(figsize=(12, 9))

        plt.subplot(231)
        plt.imshow(img_flair[:, :,n_slice], cmap='gray')
        plt.title(f'Image flair')
        plt.subplot(232)
        plt.imshow(img_t1[:,:, n_slice], cmap='gray')
        plt.title('Image t1')
        plt.subplot(233)
        plt.imshow(img_tice[:, :,n_slice], cmap='gray')
        plt.title('Image t1ce')
        plt.subplot(234)
        plt.imshow(img_t2[:, :, n_slice], cmap='gray')
        plt.title('Image t2')
        plt.subplot(235)
      #  plt.imshow(img_mask[:, :, n_slice])
      #   plt.imshow(img_mask[0,:, :, n_slice], cmap= 'summer')
      #   plt.imshow(np.ma.masked_where(img_mask[1,:, :, n_slice] == False, img_mask[1,:, :, n_slice]), cmap= 'rainbow', alpha=0.6)
      #   plt.imshow(np.ma.masked_where(img_mask[2,:, :, n_slice] == False, img_mask[2,:, :, n_slice]), cmap='winter', alpha=0.6)
        plt.title('Mask')

        plt.savefig(
                os.path.join(self.matplotimg, f'TASK1_train_{case}' + '.jpg'))


    def preprocess_mask(self, mask):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 3] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 3] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 3] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


    def img_figshow(self, image, mask, is_combined=False):
        '''

        :param image: The numpy array of one image
        :param mask : The numpy array of the segmentation to the target case
        :param is_combined : A boolean value for getting the fig in case of combined channel
        :return: To get the image figure
        '''
        leng = image.shape[2]
        print("Number of frames: \n", leng)
        if is_combined:
            for ind in range(0, 3):
                fig = plt.figure(figsize=(30,30))
                c = 1
                for frame_idx in range(leng):
                    ax = fig.add_subplot(leng // 10 + 1, 10, c)
                    ax.imshow(image[:, :, frame_idx, ind], cmap='gray')
                    c += 1
                    plt.axis('off')

                fig.tight_layout()
                plt.savefig(os.path.join(self.matplotimg,'TASK1_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'))

        else:
            fig = plt.figure(figsize=(30, 30))
            c = 1
            for frame_idx in range(leng):
                ax = fig.add_subplot(leng // 10 + 1, 10, c)
                ax.imshow(image[:, :, frame_idx], cmap='gray')
                c += 1
                plt.axis('off')
            fig.tight_layout()
            plt.savefig(
                os.path.join(self.matplotimg,'TASK1_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'))

        fig = plt.figure(figsize=(30, 30))
        c = 1
        for frame_idx in range(leng):
            ax = fig.add_subplot(leng // 10 + 1, 10, c)
            ax.imshow(mask[:, :, frame_idx])
            c += 1
            plt.axis('off')

        fig.tight_layout()
        plt.savefig(
            os.path.join(self.matplotimg, 'TASK1_Seg' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'))

    def img_combined(self, target_id, combined_x, mask, get_image_plt=True):
        # Cropping the images
        #combined_x = combined_x[56:184, 56:184, 13:141]
        #mask = mask[56:184, 56:184, 13:141]

        combined_x = combined_x[56:184, 56:184]
        mask = mask[56:184, 56:184]


        if get_image_plt:
            self.img_figshow(combined_x, mask, is_combined=True)

        val, count = np.unique(mask, return_counts=True)
        if (1 - (count[0] / count.sum())) > 0.01:
            mask = to_categorical(mask, num_classes=4)
            np.save(os.path.join(self.combinedimgs, 'images', 'BraTS2021_' + target_id + 'image.npy'), combined_x)
            np.save(os.path.join(self.combinedimgs, 'masks', 'BraTS2021_' + target_id + 'mask.npy'), mask)
        else:
            print("Image deleted")

    def get_combined_imgs(self):
        t1ce_list = sorted(glob.glob(os.path.join(self.root, '*', '*_t1ce.nii.gz')))
        flair_list = sorted(glob.glob(os.path.join(self.root, '*', '*_flair.nii.gz')))
        mask_list = sorted(glob.glob(os.path.join(self.root, '*', '*_seg.nii.gz')))

        print(len(t1ce_list))

        for img in range(len(t1ce_list)):
            print("Now preparing image and masks number: ", img)
            patient_id = flair_list[img].split("\\")[-1].split('_')[1]
            images = [flair_list[img], t1ce_list[img]]
            mask = mask_list[img]
            file_data = self.read_file(images, mask)
            self.img_combined(patient_id, file_data['combined_channel'], file_data['segmentation'], get_image_plt=False)

    def get_imgs_validation(self, path):
        t2_list = sorted(glob.glob(os.path.join(path, '*', '*_t2.nii.gz')))
        t1ce_list = sorted(glob.glob(os.path.join(path, '*', '*_t1ce.nii.gz')))
        flair_list = sorted(glob.glob(os.path.join(path, '*', '*_flair.nii.gz')))

        for img in range(len(t2_list)):
            print("Now preparing image and masks number:", img)
            patient_id = flair_list[img].split("\\")[-1].split('_')[1]
            images = [flair_list[img], t1ce_list[img], t2_list[img]]
            file_data = self.read_file(images)
            combined_x = file_data['combined_channel']
            combined_x = combined_x[56:184, 56:184, 13:141]
            np.save(os.path.join(self.validation, 'BraTS2021_' + patient_id + 'image.npy'), combined_x)

    # def splitfolders(self):
    #     input_dir = self.combinedimgs
    #     output_dir = self.output_dir
    #
    #     splitfolders.ratio(input_dir, output=output_dir, seed=42, ratio=(.75, .25), group_prefix=None)






new_obj = task1_utils(normalize=True)
#new_obj.get_images(r"D:\Personal_User\BITS\third year\Design Project\TASK1\Dataset\train\BraTS2021_00003")
target_cases = os.listdir(new_obj.root)[0:3]
count = 0
for case in target_cases:
    flair,t1,ce,t,seg,percen = new_obj.read_file(case)
    # if percen > 0.01:
    #     print(case)
    #     new_obj.get_images(flair,t1,ce,t,seg,case)
    #     count+= 1

print(count/30)
# new_obj.read_file("BraTS2021_00000", True)
