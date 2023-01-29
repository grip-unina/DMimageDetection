#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import numpy as np
import torch
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image
from io import BytesIO
import numbers


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size[1], img.size[0]
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def get_list_norm(norm_type):
    transforms_list = list()
    if norm_type == 'resnet':
        print('normalize RESNET')

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))
    elif norm_type == 'clip':
        print('normalize CLIP')
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                    std=(0.26862954, 0.26130258, 0.27577711)))
    elif norm_type == 'spec':
        print('normalize SPEC')

        transforms_list.append(normalization_fft)
        transforms_list.append(transforms.ToTensor())

    elif norm_type == 'residue3':
        print('normalize Residue3')

        transforms_list.append(normalization_residue3)

    elif norm_type == 'none':
        print('normalize 0,1')

        transforms_list.append(transforms.ToTensor())

    elif norm_type == 'xception':
        print('normalize -1,1')

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5]))

    elif norm_type == 'cooc':
        print('normalize COOC')

        transforms_list.append(normalization_cooc)

    elif norm_type == 'Nataraj2019':
        print('normalize Nataraj2019')

        transforms_list.append(normalization_Nataraj2019)
    else:
        assert False

    return transforms_list


def normalization_fft(pic):

    im = np.float32(deepcopy(np.asarray(pic))) / 255.0

    for i in range(im.shape[2]):
        img = im[:, :, i]
        fft_img = np.fft.fft2(img)
        fft_img = np.log(np.abs(fft_img) + 1e-3)
        fft_min = np.percentile(fft_img, 5)
        fft_max = np.percentile(fft_img, 95)
        if (fft_max - fft_min) <= 0:
            print('ma cosa...')
            fft_img = (fft_img - fft_min) / ((fft_max - fft_min)+np.finfo(float).eps)
        else:
            fft_img = (fft_img - fft_min) / (fft_max - fft_min)
        fft_img = (fft_img - 0.5) * 2
        fft_img[fft_img < -1] = -1
        fft_img[fft_img > 1] = 1
        im[:, :, i] = fft_img

    return im


def normalization_residue3(pic, flag_tanh=False):

    x = np.float32(deepcopy(np.asarray(pic))) / 32
    wV = (-1 * x[1:-3, 2:-2, :] + 3 * x[2:-2, 2:-2, :] - 3 * x[3:-1, 2:-2, :] + 1 * x[4:, 2:-2, :])
    wH = (-1 * x[2:-2, 1:-3, :] + 3 * x[2:-2, 2:-2, :] - 3 * x[2:-2, 3:-1, :] + 1 * x[2:-2, 4:, :])
    ress = np.concatenate((wV, wH), -1)
    if flag_tanh:
        ress = np.tanh(ress)

    ress = torch.from_numpy(ress).permute(2, 0, 1).contiguous()

    return ress


def normalization_cooc(pic):
    x = deepcopy(np.asarray(pic))
    y = x[1:, 1:, :]
    x = x[:-1, :-1, :]
    bins = np.arange(257)
    H = np.stack([np.histogram2d(x[:, :, i].flatten(), y[:, :, i].flatten(), bins, density=True)[0]
                  for i in range(x.shape[2])], 0)
    H = torch.from_numpy(H)
    return H


def normalization_Nataraj2019(pic):
    x = np.asarray(pic)
    from skimage.feature import greycomatrix
    comtx0 = greycomatrix(x[:, :, 0], [5], [0], 256, symmetric=True, normed=True).squeeze()
    comtx1 = greycomatrix(x[:, :, 1], [5], [0], 256, symmetric=True, normed=True).squeeze()
    comtx2 = greycomatrix(x[:, :, 2], [5], [0], 256, symmetric=True, normed=True).squeeze()

    ret = np.stack((comtx0, comtx1, comtx2), 0)
    return torch.from_numpy(ret).float().contiguous()


class CenterCropNoPad():
    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        h, w = img.size[1], img.size[0]
        if max(h, w) > self.siz:
            img = center_crop(img, self.siz)
        return img


class SquareCrop2p():
    def __call__(self, img):
        sizm = min(img.size[1], img.size[0])
        if sizm == 624:
            return img
        siz = 32
        while (2*siz) <= sizm:
            siz = 2*siz
        return center_crop(img, siz)


class PilRescale():
    def __init__(self, factor, interp=Image.BILINEAR):
        self.factor = factor
        self.interp = interp

    def __call__(self, img):
        h, w = img.size[1], img.size[0]
        if self.factor == 1.0:
            return img
        (width, height) = (int(img.width * self.factor), int(img.height * self.factor))
        img = img.resize((width, height), self.interp)
        return img


class PilRotate():
    def __init__(self, angle, interp=Image.BILINEAR):
        self.angle = angle
        self.interp = interp

    def __call__(self, img):
        return img.rotate(self.angle, self.interp, expand=0)


class PilResize():
    def __init__(self, target_w, interp=Image.BICUBIC):
        self.target_w = target_w
        self.interp = interp

    def __call__(self, img):
        if self.target_w == 0:
            return img
        (width, height) = (self.target_w, img.height * self.target_w // img.width)
        img = img.resize((width, height), self.interp)
        return img


class PilJpeg():
    def __init__(self, target_qf):
        self.target_qf = target_qf

    def __call__(self, img):
        if self.target_qf > 100:
            return img

        with BytesIO() as out:
            img.save(out, format='jpeg', quality=self.target_qf)
            img = Image.open(out)
            img.load()

        return img
