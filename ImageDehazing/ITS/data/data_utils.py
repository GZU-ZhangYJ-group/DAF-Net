import os
import random

import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image

from torchvision.transforms import functional as FF

norm = False


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train=False, size=None, _format='.png', only_h_flip=False, is_test=False):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.is_test = is_test
        self._format = _format
        self.only_h_flip = only_h_flip
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'gt')

    def __getitem__(self, index):
        if index < 0 or index >= len(self.haze_imgs):
            raise ValueError(f"Invalid index: {index}. Dataset size: {len(self.haze_imgs)}")
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, tuple):
            while haze.size[0] < self.size[0] or haze.size[1] < self.size[1]:
                index = random.randint(0, len(self.haze_imgs) - 1)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        if self._format:
            if self._format == '_outdoor_GT.':
                id = img.split(f'/')[-1].split('_')[0]
                fix = img.split(f'/')[-1].split('.')[-1]
                clear_name = id + self._format + fix
            else:
                id = img.split(f'/')[-1].split('_')[0]
                clear_name = id + self._format
        else:
            clear_name = img.split('/')[-1]

        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size[0], self.size[1]))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        if self.is_test:
            name = self.haze_imgs_dir[index]
            return haze, clear, name
        return haze, clear

    def augData(self, haze, clear):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze = tfs.RandomHorizontalFlip(rand_hor)(haze)
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot and not self.only_h_flip:
                haze = FF.rotate(haze, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)
        haze = tfs.ToTensor()(haze)
        if norm:
            haze = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(haze)
        clear = tfs.ToTensor()(clear)
        return haze, clear

    def __len__(self):
        #print(len(self.haze_imgs))
        return len(self.haze_imgs)


def train_dataloader(path, batch_size=64, num_workers=0, train_data='ITS'):
    image_dir = os.path.join(path, train_data)
    corp_size = (256, 256)
    if train_data == 'OTS-train':
        only_h_flip = True
        _format = '.jpg'
    elif train_data == 'O-train':
        only_h_flip = False
        corp_size = (512, 512)
        _format = '_outdoor_GT.'
    elif train_data == 'Dense-train' or train_data == 'NH-train':
        only_h_flip = False
        _format = '_GT.png'
        corp_size = (512, 512)
        #corp_size = (800,1200)
        #corp_size = (1024,1024)
    elif train_data == 'Haze6K-train':
        only_h_flip = False
        _format = None

    else:
        only_h_flip = False
        _format = '.png'

    dataloader = data.DataLoader(
        RESIDE_Dataset(image_dir, size=corp_size, train=True, _format=_format, only_h_flip=only_h_flip),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0, valid_data='SOTS-IN'):
    image_dir = os.path.join(path, valid_data)
    corp_size = 'whole img'
    if valid_data == 'Dense-test' or valid_data == 'NH-test':
        _format = '_GT.png'
    elif valid_data == 'OTS-test':  # or valid_data == 'Haze6K-test':
        _format = '.jpg'

    elif valid_data == 'O-test':
        _format = '_outdoor_GT.'

    elif valid_data == 'Haze6K-test':
        _format = None
    else:
        _format = '.png'

    dataloader = data.DataLoader(
        RESIDE_Dataset(image_dir, size=corp_size, train=False, _format=_format),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0, test_data='SOTS-IN'):
    image_dir = os.path.join(path, test_data)
    if test_data == 'Dense-test':
        _format = '_GT.png'
        corp_size = 'whole img'
    else:
        _format = '.png'
        corp_size = 'whole img'

    dataloader = data.DataLoader(
        RESIDE_Dataset(image_dir, size=corp_size, train=False, _format=_format, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
