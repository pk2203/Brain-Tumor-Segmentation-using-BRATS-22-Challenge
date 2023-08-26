import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
import pydicom
from PIL import Image

matplotlib.rcParams['animation.html'] = 'jshtml'

from main import task2_utils


class image_modf:
    def __init__(self):
        self.seq_types = {'FLAIR': [], 'T1w': [], 'T1wCE': [], 'T2w': []}
        self.matplotdir = r'C:\Users\flat2\PycharmProjects\BraTS\matplot'
        self.scan_num = {'FLAIR': 0, 'T1w': 1, 'T1wCE': 2, 'T2w': 3}
        self.main_utils = task2_utils(32)
        self.img_dir_db = self.main_utils.create_database()

    #  self.utils = main_utils.__init__(32,[256,256])
    def figshow_img(self, data):
        final_img = Image.fromarray(data)
        final_img.save(os.path.join(self.matplotdir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'))

    def show_animation(self, images):
        '''

        :param images: List of images to be animated
        :param scan_type: to get the particular scan type
        :return: matplotlib.rcParams['animation.html'] = 'jshtml'
        '''

        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        im = plt.imshow(images[0], cmap='gray')

        def animate_func(i):
            im.set_array(images[i])
            return [im]

        animation = matplotlib.animation.FuncAnimation(fig, animate_func, frames=len(images), interval=20)
        animation.save(os.path.join(self.matplotdir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + './a.mp4'))

    def scan_sequence(self, patientid, scan_type, get_mgmt=True):
        '''

        :param patientid: Patient number ID from train label csv file
        :param get_mgmt: to check the ground truth value of mgmt indicator
        :param scan_type: FLAIR, T1w, T1wCE, T2w
        :return: tight layout figure
        '''

        brats_id = self.img_dir_db[self.img_dir_db['BRATS_id'] == patientid].BraTSID_full.item()
        print(brats_id)
        path, img_array = self.main_utils.jpg_imgs(brats_id, scan_type)
        leng = len(img_array)
        print('No of images:', leng)
        if get_mgmt: print('MGMT: ', self.img_dir_db[self.img_dir_db['BRATS_id'] == patientid].MGMT_values)

        fig = plt.figure(figsize=(30, 30))
        c = 1
        for image in img_array:
            ax = fig.add_subplot(leng // 10 + 1, 10, c)
            ax.imshow(image,cmap='gray')

            c += 1

            plt.axis('off')

        fig.tight_layout()
        plt.savefig(os.path.join(self.matplotdir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.jpg'))
        return img_array

    def get_img_panel(self, data):
        '''
        :param data: the dicom data from an image
        Returns the MRI's plane from the dicom data.
        '''
        x1, y1, _, x2, y2, _ = [round(j) for j in data.ImageOrientationPatient]
        cords = [x1, y1, x2, y2]

        if cords == [1, 0, 0, 0]:
            return 'coronal'
        if cords == [1, 0, 0, 1]:
            return 'axial'
        if cords == [0, 1, 0, 0]:
            return 'sagittal'

    def obtain_middle_image(self, path, scan, plane=False):
        '''
        :param path: the path of the target case to obtain the image array
        :param scan: the main type of modality for sequence
        :param plane: == True, then returns the plane
        :return: Gives the middle image from the plane
        '''

        image_dirs, img_data = self.main_utils.jpg_imgs(path, scan)
        img = image_dirs[len(image_dirs) // 2]
        img_path = os.path.join(self.main_utils.train_img_folder,path,scan,img)
        dicom = pydicom.dcmread(img_path)
        img = self.main_utils.get_image(dicom)
        plane = self.get_img_panel(dicom)
        self.figshow_img(img)
        if plane:
            return img, plane
        else:
            return img
