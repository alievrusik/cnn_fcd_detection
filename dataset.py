import torch
import torch.nn as nn
import torch.utils.data as data
import os
import numpy as np
from natsort import natsorted


def default_loader(path, batch_size, nb_of_channels, h, w):
    pair = np.load(path)
    X, y = pair[:-batch_size].reshape(batch_size, nb_of_channels, w, h), pair[-batch_size:]
    return X, y


def dummy_collate(batch):
    # print(batch[0][0].shape, batch[0][1].shape)
    return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1] > 0)


class PatchTrainDataset(data.Dataset):

    @staticmethod
    def scan_dir(dir, fcd, ignore_index):
        result = []
        for f in os.listdir(dir):

            fcd_flag = (f.split('_')[0] == 'fcd') == fcd
            if ignore_index is None:
                ignore_index_flag = False
            else:
                ignore_index_flag = int(f.split('_')[1]) == int(ignore_index)

            if os.path.isdir(os.path.join(dir, f)) and fcd_flag and not ignore_index_flag:
                imgs_in_dir = natsorted(os.listdir(os.path.join(dir, f)))
                for i in range(len(imgs_in_dir)):
                    result += [os.path.join(dir, f, imgs_in_dir[i])]
        return result

    def __init__(self, image_dir, fcd, nb_of_channels, h, w, batch_size, ignore_index=None):
        """
        Creates an image reader
        :param image_dir: directory for images
        """

        self.images = PatchTrainDataset.scan_dir(image_dir, fcd, ignore_index)
        self.loader = lambda path: default_loader(path, batch_size, nb_of_channels, h, w)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X, y = self.loader(self.images[idx])
        X = torch.from_numpy(X).float()
        return X, y


class PatchValDataset(data.Dataset):

    def __init__(self, image_dir, fcd, nb_of_channels, h, w, index, nb_of_val_images, batch_size):
        """
        Creates an image reader
        :param image_dir: directory for images
        :param image_loader: the method for loading images
        """
        fcd_name = 'fcd' if fcd else 'nofcd'
        imgs_in_dir = natsorted(os.listdir(os.path.join(image_dir, f'{fcd_name}_{index}_patches')))[:nb_of_val_images]
        self.images = []
        for i in range(len(imgs_in_dir)):
            self.images += [os.path.join(image_dir, f'{fcd_name}_{index}_patches', imgs_in_dir[i])]
        self.loader = lambda path: default_loader(path, batch_size, nb_of_channels, h, w)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X, y = self.loader(self.images[idx])
        X = torch.from_numpy(X).float()
        return X, y
