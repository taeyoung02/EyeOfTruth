import os
import torch
import torch.nn as nn
import argparse
from barni import barni
from extract_data import TrainData, ValidData


def train(dataloader, epoch):
    print('[Epoch %d]' % (epoch + 1))
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    for batch_idx, data in enumerate(dataloader):
        img, labels = data[0], data[2]
        img = img.float()
        img = torch.unsqueeze(img, 1)
        # 0으로 초기화
        optimizer.zero_grad()

        # forward
        outputs = model(img)

        # backword
        loss = criterion(outputs, labels)
        loss.backward()

        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(batch_idx)
        if batch_idx % 500 == 499:
            print('batch_idx : %5d, loss: %.4f' % (batch_idx + 1, running_loss / 500))
            running_loss = 0.0


def valid(dataloader, epoch):
    classes = ('single', 'double')
    true = [0., 0.]
    total = [0., 0.]
    acc = [0., 0.]
    with torch.no_grad(): # 학습진행 x
        for data in dataloader:
            img, labels = data[0], data[2]
            img = img.float()
            img = torch.unsqueeze(img, 1)

            # feed forward
            outputs = model(img)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(32):
                label = labels[i]
                true[label] += c[i].item()
                total[label] += 1

    for i in range(2):
        acc[i] = 100 * true[i] / total[i]
        print('%5s accuracy : %.4f' % (
            classes[i], acc[i]))

    average_acc = sum(acc) / 2
    print('average accuracy : %.4f %%' % (average_acc))

    # save best model
    if average_acc > valid_best['average_acc']:
        torch.save(model.state_dict(), './model/barni.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='C:/') # path 선택
    args = parser.parse_args()

    data_path = args.data_path
    trainset = TrainData(data_path)
    validset = ValidData(data_path)

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True, drop_last=True)

    model = barni()
    optimizer = torch.optim.Adam(model.parameters())

    valid_best = dict(epoch=0, single_acc=0, double_Acc=0, average_acc=0)

    for epoch in range(3):
        model.train()
        train(train_dataloader, epoch)
        model.eval()
        valid(valid_dataloader, epoch)
