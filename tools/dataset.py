from torch.utils.data import Dataset, get_worker_info
import os
from multiprocessing import Pool
import glob
from tqdm import tqdm
from utils.img_utils import imread, Imread_Modcrop
import random
import numpy as np

class SRDataset(Dataset):
    @classmethod
    def init_from_dict(cls, dictionary):
        for key, value in dictionary.items():
            setattr(cls, key, value)

    @staticmethod
    def get_names(folder_list):
        all_img_names = []
        for folder_ in folder_list:
            img_names = sorted(glob.glob(folder_ + '/*.png') + glob.glob(folder_ + '/*.jpg') + glob.glob(folder_ + '/*.avif'))
            all_img_names += img_names
        return all_img_names

    @staticmethod
    def augment_image(img, trans):
        img_aug = img
        if trans == 0:
            img_aug = np.rot90(img_aug, 0)
        elif trans == 1:
            img_aug = np.rot90(img_aug, 1)
        elif trans == 2:
            img_aug = np.rot90(img_aug, 2)
        elif trans == 3:
            img_aug = np.rot90(img_aug, 3)
        elif trans == 4:
            img_aug = np.rot90(img_aug, 0)
            img_aug = np.flip(img_aug, 0)
        elif trans == 5:
            img_aug = np.rot90(img_aug, 0)
            img_aug = np.flip(img_aug, 1)
        elif trans == 6:
            img_aug = np.rot90(img_aug, 1)
            img_aug = np.flip(img_aug, 0)
        elif trans == 7:
            img_aug = np.rot90(img_aug, 1)
            img_aug = np.flip(img_aug, 1)
        return img_aug

    def load_HR(self):
        if self.pool_hr:
            pool = Pool(processes=os.cpu_count())
            for i, val in tqdm(enumerate(pool.imap(self.imread_modcrop, self.hr_names), 0),
                               desc="Preloading HR images",
                               total=len(self.hr_names)):
                self.HR_images[i] = val
            pool.close()
            pool.join()
        else:
            for i in tqdm(range(len(self.hr_names)), desc="Preloading HR images"):
                self.HR_images[i] = self.imread_modcrop(self.hr_names[i])

    def load_LR(self):
        if self.pool_lr:
            pool = Pool(processes=os.cpu_count())
            for i, val in tqdm(enumerate(pool.imap(imread, self.lr_names), 0),
                               desc="Preloading LR images",
                               total=len(self.lr_names)):
                self.LR_images[i] = val
            pool.close()
            pool.join()
        else:
            for i in tqdm(range(len(self.lr_names)), desc="Preloading LR images"):
                self.LR_images[i] = imread(self.lr_names[i])

    def __init__(self, conf_data):
        super(SRDataset, self).__init__()

        self.lr_path = conf_data['lr_path']
        self.hr_path = conf_data['hr_path']
        self.patch_cropsize = conf_data['patch_cropsize']
        self.augment = conf_data['augment']
        self.pool_lr = conf_data['pool_lr']
        self.pool_hr = conf_data['pool_hr']
        self.is_train = conf_data['is_train']
        self.scale = int(conf_data['scale'])

        self.hr_names = self.get_names(self.hr_path)
        self.lr_names = self.get_names(self.lr_path)

        self.HR_images = [None] * len(self.hr_names)
        self.LR_images = [None] * len(self.lr_names)

        self.number_of_images = len(self.hr_names)

        self.imread_modcrop = Imread_Modcrop(mod=self.scale)

        self.load_LR()
        self.load_HR()

    def __getitem__(self, item):
        return_dict = {}

        # Get idx according to the workers
        if not self.is_train:
            idx = item % self.number_of_images
        else:
            worker_info = get_worker_info()
            if worker_info is not None:
                item = random.randint(0, (self.number_of_images // worker_info.num_workers) - 1)
                idx = (item % (self.number_of_images // worker_info.num_workers)) + worker_info.id * (
                        self.number_of_images // worker_info.num_workers)
            else:
                item = random.randint(0, self.number_of_images)
                idx = item % self.number_of_images

        img_hr = self.HR_images[idx]
        img_lr = self.LR_images[idx]

        hr_dim = img_hr.shape
        if self.patch_cropsize is not False:
            i = int((hr_dim[0] - self.patch_cropsize) * random.random() // self.scale) * self.scale
            j = int((hr_dim[1] - self.patch_cropsize) * random.random() // self.scale) * self.scale
            i_lr = i // self.scale
            j_lr = j // self.scale
            img_hr = (img_hr[i:i + self.patch_cropsize,
                      j:j + self.patch_cropsize,
                      :])
            img_lr = (img_lr[i_lr:i_lr + self.patch_cropsize // self.scale,
                      j_lr:j_lr + self.patch_cropsize // self.scale,
                      :])

        if self.augment:
            t = int(8 * random.random())
            img_hr = self.augment_image(img_hr, t)
            img_lr = self.augment_image(img_lr, t)

        return_dict['img_lr'] = np.transpose(img_lr, (2, 0, 1)).copy()
        return_dict['img_hr'] = np.transpose(img_hr, (2, 0, 1)).copy()

        return return_dict

    def __len__(self):
        return len(self.hr_names)
