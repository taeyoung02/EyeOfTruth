# test other images that not in data set

from DoubleJpeg import DoubleJpeg
import torch
import glob
import numpy as np
from PIL import Image
from PIL import JpegImagePlugin



'''
preprocessing
'''
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

def convert_dict_Q_table(Q_table): # zigzag scanning
    Q_table = [Q_table[key] for key in range(len(Q_table)) if key in Q_table]
    for idx, table in enumerate(Q_table):
        Q_table[idx] = [table[i] for i in zigzag_index]
    return Q_table

def read_q_table(file_name):
    jpg = JpegImagePlugin.JpegImageFile(file_name)
    Q_table = convert_dict_Q_table(jpg.quantization)[0]
    Q_table_2d = np.zeros((8, 8))

    Q_table_idx = 0
    for i in range(0, 8):
        for j in range(0, 8):
            Q_table_2d[i, j] = Q_table[Q_table_idx]
            Q_table_idx = Q_table_idx + 1

    return Q_table_2d


class TestData(torch.utils.data.Dataset): # An abstract class representing a Dataset
    def __init__(self, data_path):
        doubles = []
        singles = []

        # load double
        double_path = data_path + '/double/'
        doubles_ = glob.glob(double_path + '*.jpg')
        doubles.extend(doubles_)

        # load single
        single_path = data_path + '/single/'
        singles_ = glob.glob(single_path + '*.jpg')
        singles.extend(singles_)

        double_label = [1] * len(doubles)
        single_label = [0] * len(singles)

        self.file_list = doubles + singles
        self.label_list = double_label + single_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx): # get q_vector
        im = Image.open(self.file_list[idx])
        im = im.convert('YCbCr')
        Y = np.array(im)[:, :, 0]

        q_table = read_q_table(self.file_list[idx])
        q_vector = q_table.flatten()
        label = self.label_list[idx]
        return (Y, q_vector, label)


'''
predict
'''
def test(dataloader, epoch):
    classes = ('single', 'double')
    class_correct = [0.,0.]
    class_total =[0.,0.]
    class_acc = [0.,0.]

    with torch.no_grad():
        count=0
        for samples in dataloader:
            count+=1
            print(count)
            Ys, qvectors, labels = samples[0], samples[1], samples[2]
            Ys = Ys.float()
            Ys = torch.unsqueeze(Ys, axis=1)
            qvectors = qvectors.float()

            # feed forward
            outputs = net(Ys, qvectors)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            print(predicted)
            if c==1:
                print("collect")




data_path = 'testset/'
test_dataset = TestData(data_path)
print(test_dataset)

net = DoubleJpeg()

#load weights
net.load_state_dict(torch.load('./model/best.pth'))

optimizer = torch.optim.Adam(net.parameters())

test_dataloader = torch.utils.data.DataLoader(test_dataset)
net.eval()
test(test_dataloader, epoch=0)