import bisect
import os
import cv2
import numpy as np
import albumentations
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
#import put.PUT.image_synthesis.flow_viz


def make_colorwheel():


    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


class ImagePaths(Dataset):
    def __init__(self, paths, sketch_paths=None, segmentation_paths=None, labels=None):

        self.labels = dict() if labels is None else labels
        self.labels["abs_path"] = paths
        if segmentation_paths is not None:
            self.labels["segmentation_path"] = segmentation_paths
        if sketch_paths is not None:
            self.labels["sketch_path"] = sketch_paths
        # self._length = len(paths)
        self.valid_index = list(range(len(paths)))

    def __len__(self):
        # return self._length
        return len(self.valid_index)

    def _read_image(self, image_path, type='image'):
        #根据image_path打开.flo文件
        if image_path.endswith('.flo'):
            with open(image_path, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (h[0], w[0], 2))

            #把data2D变成3通道的numpy，其中多出来的通道为第一个通道的值
            image = flow_compute_color(data2D[:, :, 0], data2D[:, :, 1])
        # image = Image.open(image_path)
        # if type == 'image':
        #     if not image.mode == "RGB":
        #         image = image.convert("RGB")
        #     image = np.array(image).astype(np.float32) # H x W x 3
        # elif type in ['segmentation_map   ', 'sketch_map']:
        #     image = np.array(image).astype(np.float32) # H x W
        # else:
        #     raise NotImplementedError
        return image
    

    def __getitem__(self, idx):
        i = self.valid_index[idx]
        #2024/8/5 返回图片、原flow，可视化图，归一化参数
        image = self._read_image(self.labels["abs_path"][i], type='image')
        example = {
            'image': image,
        }
        if 'segmentation_path' in self.labels:
            seg_map = self._read_image(self.labels["segmentation_path"][i], type='segmentation_map')
            example['segmentation_map'] = seg_map
        if 'sketch_path' in self.labels:
            ske_map = self._read_image(self.labels["sketch_path"][i], type='sketch_map')  
            example['sketch_map'] = ske_map  
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    
    def remove_files(self, file_path_set):
        if not isinstance(file_path_set, set):
            file_path_set = set(file_path_set)
        valid_index = []
        for i in range(len(self.labels['abs_path'])):
            # import pdb; pdb.set_trace()
            p = self.labels['abs_path'][i]
            if p not in file_path_set:
                valid_index.append(i)
        print('remove {} files, current {} files'.format(len(self.valid_index)-len(valid_index), len(valid_index)))
        self.valid_index = valid_index
            