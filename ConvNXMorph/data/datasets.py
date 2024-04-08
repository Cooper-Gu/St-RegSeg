import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torch.utils.data as Data

def Readimage(image):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(image))
    return arr

class Dataset(Data.Dataset):
    def __init__(self,list_filename,fixed_1_path, fixed_2_path, moving_path, transforms):
        # 初始化
        self.list_filename = list_filename
        self.fixed_1_path = fixed_1_path
        self.fixed_2_path = fixed_2_path
        self.moving_path = moving_path
        self.transforms = transforms

    def __len__(self):
        # 返回数据集的大小
        return len(self.list_filename)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        filename = self.list_filename[index]
        fixed_image_1_path = os.path.join(self.fixed_1_path, filename)
        fixed_image_2_path = os.path.join(self.fixed_2_path, filename)
        moving_image_path = os.path.join(self.moving_path, filename)

        fixed_1_image = Readimage(fixed_image_1_path)[np.newaxis, ...]
        fixed_2_image = Readimage(fixed_image_2_path)[np.newaxis, ...]
        moving_image = Readimage(moving_image_path)[np.newaxis, ...]
        fixed_1_image = self.transforms(fixed_1_image)
        fixed_2_image = self.transforms(fixed_2_image)
        moving_image = self.transforms(moving_image)
        # 返回值自动转换为torch的tensor类型

        return moving_image,fixed_1_image, fixed_2_image

class TestDataset(Data.Dataset):
    def __init__(self,list_filename,fixed_path,moving_path, transforms):
        # 初始化
        self.list_filename = list_filename
        self.fixed_path = fixed_path
        self.moving_path = moving_path
        self.transforms = transforms

    def __len__(self):
        # 返回数据集的大小
        return len(self.list_filename)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        filename = self.list_filename[index]
        fixed_image_path = os.path.join(self.fixed_path, filename)
        moving_image_path = os.path.join(self.moving_path, filename)

        fixed_image = Readimage(fixed_image_path)[np.newaxis, ...]
        moving_image = Readimage(moving_image_path)[np.newaxis, ...]
        fixed_image = self.transforms(fixed_image)
        moving_image = self.transforms(moving_image)
        # 返回值自动转换为torch的tensor类型

        return moving_image,fixed_image,filename

class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

