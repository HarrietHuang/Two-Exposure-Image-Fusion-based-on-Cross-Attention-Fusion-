from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
# from utils import is_image_file, load_img
import os
import numpy as np
import cv2
import argparse
import glob
from PIL import Image
from torchvision import transforms as T
from torchvision import transforms
import IPython.display as display
from helper import cal_TMQI
import torch.utils.data as data
import torch
import numpy as np
from skimage.transform import pyramid_gaussian, resize
# teacher
import numpy as np
from skimage.transform import pyramid_gaussian
import cv2
from scipy import signal
import sys
# from cv2.ximgproc import guidedFilter
import random
from skimage.color import rgb2ycbcr
from skimage.transform import pyramid_gaussian, resize
from skimage.io import imsave
from skimage import img_as_ubyte
import torchvision.transforms.functional as TF
from helper import savefig
def adjust_gamma(image, gamma=1):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def upsample(mono):
    img_shape = mono.shape
    if len(img_shape) == 3 and img_shape[2] != 1:
        sys.exit('failure - upsample')

    C = np.zeros([img_shape[0] * 2, img_shape[1] * 2])
    C[1::2, 1::2] = mono
    t1 = list([[0.1250, 0.5000, 0.7500, 0.5000, 0.1250]])
    t2 = list([[0.1250], [0.5000], [0.7500], [0.5000], [0.1250]])
    myj = signal.convolve2d(C, t1, mode="same")
    myj = signal.convolve2d(myj, t2, mode="same")
    return myj


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def la_filter(mono):
    img_shape = mono.shape
    C = np.zeros(img_shape)
    t1 = list([[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]])
    # for i in range(0, img_shape[0]):
    #     for j in range(0, img_shape[1]):
    #         C[i, j] = abs(np.sum(mono[i:i + 3, j:j + 3] * t1))
    myj = signal.convolve2d(mono, t1, mode="same")
    return myj


def contrast(I, exposure_num, img_rows, img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        mono = rgb2gray(I[i])
        C[:, :, i] = la_filter(mono)

    return C


def saturation(I, exposure_num, img_rows, img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = I[i][:, :, 0]
        G = I[i][:, :, 1]
        B = I[i][:, :, 2]
        mu = (R + G + B) / 3
        C[:, :, i] = np.sqrt(
            ((R - mu) ** 2 + (G - mu) ** 2 + (B - mu) ** 2) / 3)
    return C


def well_exposedness(I, exposure_num, img_rows, img_cols):
    sig = 0.2
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = np.exp(-.4 * (I[i][:, :, 0] - 0.5) ** 2 / sig ** 2)
        G = np.exp(-.4 * (I[i][:, :, 1] - 0.5) ** 2 / sig ** 2)
        B = np.exp(-.4 * (I[i][:, :, 2] - 0.5) ** 2 / sig ** 2)
        C[:, :, i] = R * G * B
    return C


def gaussian_pyramid(I, nlev, multi):
    pyr = []

    # for ii in range(0,nlev):
    #     temp = pyramid_gaussian(I, downscale=2)
    #     pyr.append(temp)
    for (i, resized) in enumerate(pyramid_gaussian(I, downscale=2, multichannel=multi)):
        if i == nlev:
            break
        pyr.append(resized)
    return pyr


def laplacian_pyramid(I, nlev, mult=True):
    pyr = []
    expand = []
    pyrg = gaussian_pyramid(I, nlev, multi=mult)
    for i in range(0, nlev - 1):

        # expand_temp = cv2.resize(pyrg[i + 1], (pyrg[i].shape[1],
        # pyrg[i].shape[0]))
        expand_temp = resize(
            pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]), preserve_range=True, anti_aliasing=False)
        # expand_temp = resize(pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]), preserve_range=True, anti_aliasing=False)
        # for j in range(3):
        # expand_temp[:,:,j] = upsample(pyrg[i+1][:,:,j])
        temp = pyrg[i] - expand_temp
        expand.append(expand_temp)
        pyr.append(temp)
    pyr.append(pyrg[nlev - 1])
    expand.append(pyrg[nlev - 1])
    return pyr, expand


def reconstruct_laplacian_pyramid(pyr):
    nlev = len(pyr)
    # print('nlev', nlev)
    R = pyr[nlev - 1]
    for i in range(nlev - 2, -1, -1):
        # R = pyr[i+1]
        odd = R.shape
        # print('odd ',odd)

        # C = np.zeros([odd[0]*2, odd[1]*2, odd[2]])
        # for j in range(odd[2]):
        # print('R', R.shape)
        # print('C', C.shape)
        # print('pyr', pyr[i][:,:,j].shape)
        # upsample(R[:,:,j])#
        C = pyr[i] + cv2.resize(R, (pyr[i].shape[1], pyr[i].shape[0]))
        # C[:,:,j] = pyr[i][:,:,j]  + cv2.resize(R,(pyr[i].shape[1],
        # pyr[i].shape[0]))#upsample(R[:,:,j])#
        R = C
    return R


def Gaussian1D(cen, std, YX1):
    y = np.zeros((1, YX1))
    for i in range(0, YX1):
        y[0][i] = np.exp(-((i - cen)**2) / (2 * (std**2)))
    y = np.round(y * (YX1 - 1))
    return y


def gaussian_pyramid(I, nlev, multi):
    pyr = []

    # for ii in range(0,nlev):
    #     temp = pyramid_gaussian(I, downscale=2)
    #     pyr.append(temp)
    for (i, resized) in enumerate(pyramid_gaussian(I, downscale=2, multichannel=multi)):
        if i == nlev:
            break
        pyr.append(resized)
    return pyr


def laplacian_pyramid(I, nlev, mult=True):
    pyr = []
    expand = []
    pyrg = gaussian_pyramid(I, nlev, multi=mult)
    for i in range(0, nlev - 1):

        # expand_temp = cv2.resize(pyrg[i + 1], (pyrg[i].shape[1],
        # pyrg[i].shape[0]))
        expand_temp = resize(
            pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]), preserve_range=True, anti_aliasing=False)
        temp = pyrg[i] - expand_temp
        expand.append(expand_temp)
        pyr.append(temp)
    pyr.append(pyrg[nlev - 1])
    expand.append(pyrg[nlev - 1])
    return pyr, expand


def cfusion(uexp, oexp):
    beta = 2
    vFrTh = 0.16
    RadPr = 3

    I = (uexp, oexp)
    r = uexp.shape[0]
    c = uexp.shape[1]
    n = 2
    nlev = round(np.log(min(r, c)) / np.log(2)) - beta
    nlev = int(nlev)
    RadFr = RadPr * (1 << (nlev - 1))

    W = np.ones((r, c, n))

    W = np.multiply(W, contrast(I, n, r, c))
    W = np.multiply(W, saturation(I, n, r, c))
    W = np.multiply(W, well_exposedness(I, n, r, c))

    W = W + 1e-12 # shape (512, 512, 2)
    # print(W)
    Norm = np.array([np.sum(W, 2), np.sum(W, 2)]) # shape (2, 512, 512)
    Norm = Norm.swapaxes(0, 2) # shape (512, 512, 2)
    Norm = Norm.swapaxes(0, 1) # shape (512, 512, 2)
    W = W / Norm #shape (512, 512, 2)/(512, 512, 2)

    II = (uexp / 255.0, oexp / 255.0)

    pyr = gaussian_pyramid(np.zeros((r, c, 3)), nlev, multi=True)
    for i in range(0, n):
        pyrw = gaussian_pyramid(W[:, :, i], nlev, multi=False)
        pyri, content = laplacian_pyramid(II[i], nlev, mult=True)
        for ii in range(0, nlev):
            w = np.array([pyrw[ii], pyrw[ii], pyrw[ii]])
            w = w.swapaxes(0, 2)
            w = w.swapaxes(0, 1)
            pyr[ii] = pyr[ii] + w * pyri[ii]
    R = reconstruct_laplacian_pyramid(pyr)
    # R = cv2.cvtColor(R.astype(np.float32), cv2.COLOR_YCR_CB2BGR)
    # R = ycbcr2rgb(R)

    # R = R * 255
    return R

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

class HDRdatasets_dynamic_compose(data.Dataset):

    def __init__(self, train=True, transforms=None):
        out_img_train = []
        gt_img_train = []
        img = []
        if train:
            for i in os.listdir(r'../random_select'):
                img_list = glob.glob(
                    r'../random_select' + '\\' + i + r'\*.JPG')
                for j in img_list:
                    # print(j)
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)
        else:
            for i in os.listdir(r'../random_select2'):
                img_list = glob.glob(
                    r'../random_select2' + '\\' + i + r'\*.JPG')
                img_list1 = glob.glob(
                    r'../random_select2' + '\\' + i + r'\*.png')
                for j in img_list:
                    # print(j)
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)

        self.train = train
        self.gt_img_train = gt_img_train
        self.out_img_train = out_img_train
        self.img = img


    def transform(self, image1, image2, mask):
        # Resize
        # if not self.train:
        # if self.train:
        # resize = transforms.Resize(size=(image1.size[0]//3, image1.size[1]//3))
        # else:s
        if self.train:
            resize = transforms.Resize(size=(512, 512))
        else :
            resize = transforms.Resize(size=(512, 512))
        # print(image1.size[0]//3, image1.size[1]//3)
        # random_crop = transforms.RandomCrop((128, 128))
        # image1 = random_crop(image1)
        # image2 = random_crop(image2)
        # mask = random_crop(mask)
        if self.train:
            # # Random crop
            # i, j, h, w = transforms.RandomCrop.get_params(
            #     image1, output_size=(128, 128))
            # image1 = TF.crop(image1, i, j, h, w)
            # image2 = TF.crop(image2, i, j, h, w)
            # mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image1 = TF.hflip(image1)
                image2 = TF.hflip(image2)
                mask = TF.hflip(mask)

        # Transform to tensor
        image1 = TF.to_tensor(image1)
        image2 = TF.to_tensor(image2)
        mask = TF.to_tensor(mask)

        image1 = TF.normalize(image1,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        image2 = TF.normalize(image2,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        mask = TF.normalize(mask,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

        return image1, image2,mask
    def __getitem__(self, index):
        #         print(index)
        #         print(self.out_img_train[index])
        augmentation = False
        filename = self.img[2 * index]
        label_path = self.gt_img_train[index]
#         out_path = self.out_img_train[index]
        img1_path = self.img[2 * index]
        img2_path = self.img[2 * index + 1]

        # print(label_path,self.out_img_train[index])

        label = Image.open(label_path).convert('YCbCr')
        # print(img1_path)
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1_ycbcr = img1.convert('YCbCr')
        img2_ycbcr = img2.convert('YCbCr')

        img1, img2, label = self.transform(img1_ycbcr, img2_ycbcr, label)

        return img1, img2, label, filename
    def __len__(self):

        return len(self.gt_img_train)


class HDRdatasets_lapacian(data.Dataset):

    def __init__(self, train=True, transforms=None):
        out_img_train = []
        gt_img_train = []
        img = []
        if train:
            for i in os.listdir(r'../random_select'):
                img_list = glob.glob(
                    r'../random_select' + '//' + i + r'/*.JPG')
                for j in img_list:
                    # print(j)
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)
        else:
            for i in os.listdir(r'../random_select2'):
                img_list = glob.glob(
                    r'../random_select2' + '//' + i + r'/*.JPG')
                img_list1 = glob.glob(
                    r'../random_select2' + '//' + i + r'/*.png')
                for j in img_list:
                    # print(j)
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)

        self.train = train
        self.gt_img_train = gt_img_train
        self.out_img_train = out_img_train
        self.img = img


    def transform(self, image1, image2, mask):
        # Resize
        # if not self.train:
        # if self.train:
        # resize = transforms.Resize(size=(image1.size[0]//3, image1.size[1]//3))
        # else:
        if self.train:
            resize = transforms.Resize(size=(512, 512))
        else :
            resize = transforms.Resize(size=(1024, 1024))
        # print(image1.size[0]//3, image1.size[1]//3)
        image1 = resize(image1)
        image2 = resize(image2)
        mask = resize(mask)
        if self.train:
            # Random crop
            # i, j, h, w = transforms.RandomCrop.get_params(
            #     image1, output_size=(224, 224))
            # image1 = TF.crop(image1, i, j, h, w)
            # image2 = TF.crop(image2, i, j, h, w)
            # mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image1 = TF.hflip(image1)
                image2 = TF.hflip(image2)
                mask = TF.hflip(mask)
        #lapacian pyramid
        pyr_im1, content_im1 = laplacian_pyramid(image1, 3)
        pyr_im2, content_im2 = laplacian_pyramid(image2, 3)
        pyr_im_gt, content_im_gt = laplacian_pyramid(mask, 3)

        # print(pyr_im1[0].shape)
        # print(pyr_im1[1].shape)
        # print(pyr_im1[2].shape)
        # print(content_im1[0].shape)
        # print(content_im1[1].shape)
        # print(content_im1[2].shape)
        pyr_im1_tensor = [TF.to_tensor(i) for i in pyr_im1]
        pyr_im2_tensor = [TF.to_tensor(i) for i in pyr_im2]
        pyr_im_gt_tensor = [TF.to_tensor(i) for i in pyr_im_gt]

        content_im1_tensor = [TF.to_tensor(i) for i in content_im1]
        content_im2_tensor = [TF.to_tensor(i) for i in content_im2]
        content_im_gt_tensor = [TF.to_tensor(i) for i in content_im_gt]
        # Transform to tensor
        pyr_im1_tensor = [TF.normalize(i,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]) for i in pyr_im1_tensor]
        pyr_im2_tensor = [TF.normalize(i,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]) for i in pyr_im2_tensor]
        pyr_im_gt_tensor = [TF.normalize(i,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]) for i in pyr_im_gt_tensor]

        content_im1_tensor = [TF.normalize(i,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]) for i in content_im1_tensor]
        content_im2_tensor = [TF.normalize(i,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]) for i in content_im2_tensor]
        content_im_gt_tensor = [TF.normalize(i,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]) for i in content_im_gt_tensor]

        return pyr_im1_tensor, pyr_im2_tensor, pyr_im_gt_tensor,  content_im1_tensor, content_im2_tensor, content_im_gt_tensor

    def __getitem__(self, index):
        #         print(index)
        #         print(self.out_img_train[index])
        augmentation = False
        filename = self.img[2 * index]
        label_path = self.gt_img_train[index]
#         out_path = self.out_img_train[index]
        img1_path = self.img[2 * index]
        img2_path = self.img[2 * index + 1]

        # print(label_path,self.out_img_train[index])

        label = Image.open(label_path).convert('YCbCr')
        # print(img1_path)
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1_ycbcr = img1.convert('YCbCr')
        img2_ycbcr = img2.convert('YCbCr')

        pyr_im1_tensor, pyr_im2_tensor, pyr_im_gt_tensor,  content_im1_tensor, content_im2_tensor, content_im_gt_tensor = self.transform(img1_ycbcr, img2_ycbcr, label)
        # print(pyr_im1_tensor[-1].shape)
        # print(pyr_im2_tensor[-1].shape)
        # print(pyr_im_gt_tensor[-1].shape)
        return pyr_im1_tensor, pyr_im2_tensor, pyr_im_gt_tensor,  content_im1_tensor, content_im2_tensor, content_im_gt_tensor, filename
    def __len__(self):

        return len(self.gt_img_train)

class HDRdatasets_other_dataset(data.Dataset):

    def __init__(self, train=True, transforms=None):
        # out_img_train = []
        gt_img_train = []
        img = []
        if train:
            for i in os.listdir(r'../random_select'):
                img_list = glob.glob(
                    r'../random_select' + '//' + i + r'/*.JPG')
                # print(img_list)
#                 img_list1 = glob.glob('./matlab_compose'+'//'+i+r'/*.png')
                for j in img_list:
                    # print(j)
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)
        else:
            for i in os.listdir(r'C:/Users/admin/Downloads/MEFDatabase/all_select'):
                img_list = glob.glob(
                    r'C:/Users/admin/Downloads/MEFDatabase/all_select' + '//' + i + r'/*.png')
                # img_list1 = glob.glob(
                    # r'C:\Users\admin\Downloads\MEFDatabase\all_select' + '\\' + i + r'\*.png')
                for j in img_list:
                    # print(j)
                    img.append(j)

        self.train = train
        self.gt_img_train = None
        # self.out_img_train = out_img_train
        self.img = img


    def transform(self, image1, image2):
        # Resize
        # if not self.train:
        # if self.train:
        # resize = transforms.Resize(size=(1024, 1024))
        # else:

        #resize = transforms.Resize(size=(image1.size[1]//3, image1.size[0]//3))
        # print(image1.size[0]//3, image1.size[1]//3)
        # image1 = resize(image1)
        # image2 = resize(image2)
        if self.train:
            # Random crop
            # i, j, h, w = transforms.RandomCrop.get_params(
            #     image1, output_size=(224, 224))
            # image1 = TF.crop(image1, i, j, h, w)
            # image2 = TF.crop(image2, i, j, h, w)
            # mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image1 = TF.hflip(image1)
                image2 = TF.hflip(image2)

        # Transform to tensor
        image1 = TF.to_tensor(image1)
        image2 = TF.to_tensor(image2)

        image1 = TF.normalize(image1,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        image2 = TF.normalize(image2,[0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

        return image1, image2
    def __getitem__(self, index):
        #         print(index)
        #         print(self.out_img_train[index])
        augmentation = False
        filename = self.img[2 * index]
        # label_path = self.gt_img_train[index]
#         out_path = self.out_img_train[index]
        img1_path = self.img[2 * index]
        img2_path = self.img[2 * index + 1]

        # print(label_path,self.out_img_train[index])

        # label = Image.open(label_path).convert('YCbCr')
        # print(img1_path)
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1_ycbcr = img1.convert('YCbCr')
        img2_ycbcr = img2.convert('YCbCr')

        img1, img2 = self.transform(img1_ycbcr, img2_ycbcr)
        print('=========================================>')
        print(img1)
        print('=========================================>')
        print(img2)
        print('=========================================>')
        print(filename)
        return img1, img2, img2, filename

    def __len__(self):

        return len(self.img)//2


class HDRDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.folder = root
        self.indexes = open(root + '\annotations.txt').read().splitlines()

    def __getitem__(self, index):
        ldr_image, hdr_image = self.indexes[index].split('\t')
        ldr_image = cv2.imread(ldr_image)
        ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)
        ldr_image = ldr_image / 255

        hdr_image = cv2.imread(hdr_image, cv2.IMREAD_ANYDEPTH)
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)

        return ldr_image.transpose(2, 0, 1), hdr_image.transpose(2, 0, 1)

    def __len__(self):
        return len(self.indexes)


def get_loader(root, batch_size, shuffle=True):
    # dataset = HDRDataset(root=root)

    # num_train = int(len(dataset) * 0.8)
    # num_val = len(dataset) - num_train
    # train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=shuffle,
    #                           drop_last=True)

    # val_loader = DataLoader(dataset=val_dataset,
    #                         batch_size=batch_size,
    #                         shuffle=shuffle,
    #                         drop_last=True)
    # T.RandomHorizontalFlip(),
    # transforms = T.Compose([T.Resize([512, 512]), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transforms = T.Compose([T.Resize([224, 224]), T.ToTensor(), torch.tensor([0.5, 0.5, 0.5]) ])
    # transforms = T.Compose([T.Resize([224, 224]),T.ToTensor(),
    # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = HDRdatasets_dynamic_compose(True)
    test_dataset = HDRdatasets_dynamic_compose(False)
    print(len(test_dataset))
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_dataloader, test_dataloader

    # return train_loader, val_loader

if __name__ == '__main__':

    transforms=T.Compose([T.Resize([512, 512]), T.ToTensor()])

    test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
    train_dataset=HDRdatasets_lapacian(True, transforms)
    train_dataloader=data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
# #     # for folder in train_dataloader:
    for pyr_im1_tensor, pyr_im2_tensor, pyr_im_gt_tensor,  content_im1_tensor, content_im2_tensor, content_im_gt_tensor, filename in train_dataloader:
        print(filename[0].split('\\')[-1])
        # print(filename[0].replace('.JPG','').split('\\')[-1].split('\\')[0])
        # break
