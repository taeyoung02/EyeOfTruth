'''
data proprecessing
'''

import torch
import glob
import numpy as np
from PIL import Image
from PIL import JpegImagePlugin

zigzag_index = (
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
)

def convert_dict_Q_tables(Q_tables): # zigzag scanning
    Q_tables = [Q_tables[key] for key in range(len(Q_tables)) if key in Q_tables]
    for idx, table in enumerate(Q_tables):
        Q_tables[idx] = [table[i] for i in zigzag_index]
    return Q_tables

def read_q_table(file_name):
    jpg = JpegImagePlugin.JpegImageFile(file_name)
    Q_table = convert_dict_Q_tables(jpg.quantization)[0]
    Q_table_2d = np.zeros((8, 8))

    Q_table_idx = 0
    for i in range(0, 8):
        for j in range(0, 8):
            Q_table_2d[i, j] = Q_table[Q_table_idx]
            Q_table_idx = Q_table_idx + 1

    return Q_table_2d # 8x8


# load single, double dataset and q_table
class TrainData(torch.utils.data.Dataset): # An abstract class representing a Dataset
    def __init__(self, data_path):
        train_dir = ['1','2','3','4']
        doubles = []
        singles = []

        # load double
        for i in train_dir:
            double_path = data_path + '/double'
            files = glob.glob(double_path + '/' + i + '/*.jpg')
            doubles.extend(files)

        # load single
        for i in train_dir:
            single_path = data_path + '/single'
            files = glob.glob(single_path + '/' + i + '/*.jpg')
            singles.extend(files)

        double_label = [1] * len(doubles)
        single_label = [0] * len(singles)

        self.file_list = doubles + singles
        self.label_list = double_label + single_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx): # get q_vector
        im = Image.open(self.file_list[idx])
        im = im.convert('YCbCr') # 이미지 색 공간 변환 RGB 색공간을 YCbCr 색공간으로 변환한다. Y성분은 픽셀의 밝기를 나타내고, Cb와 Cr은 색차성분을 나타낸다.
        img = np.array(im)[:, :, 0] # 1번째 channel만 사용 (256,256)

        q_table = read_q_table(self.file_list[idx])
        q_vector = q_table.flatten()
        label = self.label_list[idx]

        return (img, q_vector, label)


# load valid set
class ValidData(TrainData):
    def __init__(self, data_path):
        valid_dir = ['5']
        doubles = []
        singles = []

        for i in valid_dir:
            double_path = data_path + 'double'
            files = glob.glob(double_path + '/' + i + '/*.jpg')
            doubles.extend(files)

        for i in valid_dir:
            single_path = data_path + 'single'
            files = glob.glob(single_path + '/' + i + '/*.jpg')
            singles.extend(files)

        double_label = [1] * len(doubles)
        single_label = [0] * len(singles)

        self.file_list = doubles + singles
        self.label_list = double_label + single_label
