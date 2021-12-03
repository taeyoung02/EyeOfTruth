# valid set으로 평가
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from DoubleJpeg import DoubleJpeg
from extract_data import TrainData, ValidData


# vali set 평가
def valid(dataloader, epoch):
    classes = ('single', 'double')
    class_correct = [0., 0.]
    class_total = [0., 0.]
    class_acc = [0., 0.]
    with torch.no_grad():
        count = 0
        for samples in dataloader:
            count += 1
            print(count)
            Ys, qvectors, labels = samples[0], samples[1], samples[2]
            Ys = Ys.float()
            Ys = torch.unsqueeze(Ys, axis=1)
            qvectors = qvectors.float()

            # feed forward
            outputs = net(Ys, qvectors)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(32):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        class_acc[i] = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %.2f %%' % (
            classes[i], class_acc[i]))

    total_acc = (class_acc[0] + class_acc[1]) / 2
    print('Accuracy of %5s : %.2f %%' % ('Total', total_acc))

    # calculate valid best
    if total_acc > valid_best['total_acc']:
        valid_best['total_acc'] = total_acc
        valid_best['single_acc'] = class_acc[0]
        valid_best['double_acc'] = class_acc[1]
        valid_best['epoch'] = epoch + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='C:/')
    args = parser.parse_args()

    data_path = args.data_path
    valid_dataset = ValidData(data_path)

    net = DoubleJpeg()

    # load weights
    net.load_state_dict(torch.load('./model/best.pth'))

    net.to()
    optimizer = torch.optim.Adam(net.parameters())

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True, drop_last=True)

    valid_best = dict(epoch=0, single_acc=0, double_Acc=0, total_acc=0)
    net.eval()
    valid(valid_dataloader, epoch=0)
    print('Done')
