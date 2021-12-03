from __future__ import absolute_import, unicode_literals

import glob
import os
import pickle
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.generic import ListView
from .forms import UploadFileForm
from .models import Post, Category
from celery import Celery
from django.conf import settings
import sys
sys.path.append('C:\\Users\\owner\\PycharmProjects\\pythonProject1')
from DJPEG.DoubleJpeg import DoubleJpeg
import jsonpickle
import torch
import torch.nn.functional as F
import math
from PIL import JpegImagePlugin
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from DJPEG.extract_data import read_q_table
import json
#from .tasks import waiting, testing

from django.shortcuts import render
from .celery import app

@app.task
def ready():
    net = DoubleJpeg()
    # load weights
    net.load_state_dict(torch.load('./model/best.pth', map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(net.parameters())
    return net

@app.task
def _extract_patches(Y, patch_size):
    patches = list()
    h, w = Y.shape[0:2]
    H = (h - patch_size) // 32
    W = (w - patch_size) // 32
    for i in range(0, H * 32, 32):
        for j in range(0, W * 32, 32):
            patch = Y[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches, H, W

@app.task
def localizing_double_JPEG(Y, qvectors, net):
    net.eval()
    result = 0
    PATCH_SIZE = 16

    patches, H, W = _extract_patches(Y, patch_size=PATCH_SIZE)

    qvectors = torch.from_numpy(qvectors).float()
    qvectors = torch.unsqueeze(qvectors, axis=0)


    result = np.zeros((H, W))

    # import pdb; pdb.set_trace()
    num_batches = math.ceil(len(patches) / 32)

    # result = np.zeros_like(Y)

    result_flatten = np.zeros((H * W))


    for i in range(num_batches):
        if i == (num_batches - 1):  # last batch
            batch_Y = patches[i * 32:]
        else:
            batch_Y = patches[i * 32:(i + 1) * 32]
        print('[{} / {}] Detecting...'.format(i, num_batches))

        batch_size = len(batch_Y)
        batch_Y = np.array(batch_Y)
        batch_Y = torch.unsqueeze(torch.from_numpy(batch_Y).float(), axis=1)
        batch_qvectors = torch.repeat_interleave(qvectors, batch_size, dim=0)
        batch_output = net(batch_Y, batch_qvectors)
        batch_output = F.softmax(batch_output, dim=1)

        result_flatten[(i * 32):(i * 32) + batch_size] = \
            batch_output.detach().cpu().numpy()[:, 0]
    result = np.reshape(result_flatten, (H, W))

    return result



class PostList(ListView):
    model = Post
    paginate_by = 5

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(PostList, self).get_context_data(**kwargs)
        context['category_list'] = Category.objects.all()
        context['posts_without_category'] = Post.objects.filter(category=None).count()

        return context


@app.task
def convert_dict_Q_tables(Q_tables):  # zigzag scanning
    zigzag_index = (
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63,
    )
    Q_tables = [Q_tables[key] for key in range(len(Q_tables)) if key in Q_tables]
    for idx, table in enumerate(Q_tables):
        Q_tables[idx] = [table[i] for i in zigzag_index]
    return Q_tables

@app.task
def read_q_table(file_name):
    jpg = JpegImagePlugin.JpegImageFile(file_name)
    Q_table = convert_dict_Q_tables(jpg.quantization)[0]
    Q_table_2d = np.zeros((8, 8))

    Q_table_idx = 0
    for i in range(0, 8):
        for j in range(0, 8):
            Q_table_2d[i, j] = Q_table[Q_table_idx]
            Q_table_idx = Q_table_idx + 1

    return Q_table_2d  # 8x8



class TestData(torch.utils.data.Dataset): # An abstract class representing a Dataset
    def __init__(self, data_path):
        # load
        self.file_list = glob.glob('media/' + '*.jpg')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx): # get q_vector
        q_table = read_q_table(self.file_list[idx])
        im = Image.open(self.file_list[idx])
        im = im.convert('YCbCr')
        Y = np.array(im)[:, :, 0]
        
        q_vector = q_table.flatten()
        return (Y, q_vector)


'''
predict
'''
@app.task
def test(dataloader, net):
    with torch.no_grad():
        for samples in dataloader:
            Ys, qvectors = samples[0], samples[1]
            Ys = Ys.float()
            Ys = torch.unsqueeze(Ys, axis=1)
            qvectors = qvectors.float()

            # feed forward
            outputs = net(Ys, qvectors)
            _, predicted = torch.max(outputs, 1)
            print(_, predicted)
            if predicted==1:
                print("이 사진은 조작되었습니다")
                return 1
            if predicted==0:
                return 0



# def waiting(request, text):
#     context = {
#         'msg': text,
#         'manipulated': 1,
#     }
#     return render(request, 'blog/upload.html', context)

@app.task
def visualize(net):
    # visualize
    file_path = glob.glob('media/' + '*.jpg')[0]

    # read quantization table of Y channel from jpeg images
    qvector = read_q_table(file_path).flatten()

    # read an image
    im = Image.open(file_path)
    im = im.convert('YCbCr')
    Y = np.array(im)[:, :, 0]

    # load pre-trained weights
    result = localizing_double_JPEG(Y, qvector,  net)  # localizaing using trained detecting double JPEG network.
    # plot and save the result
    fig = plt.figure()
    columns = 2
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(Image.open(file_path))
    plt.title('input')

    fig.add_subplot(rows, columns, 2)
    result = result * 255
    result = result.astype('uint8')
    img_result = Image.fromarray(result)
    img_result.convert("L")
    plt.imshow(img_result, cmap='gray', vmin=0, vmax=255)
    plt.title('result')
    plt.savefig('./media/result.jpg')
    return 1


@app.task
def eyeoftruth(net):
    data_path = 'media/'
    test_dataset = TestData(data_path)

    test_dataloader = torch.utils.data.DataLoader(test_dataset)
    net.eval()
    testresult = test(test_dataloader, net)

    if testresult==1:
        return 1
    else:
        return 0




def upload_file(request):
    net = ready()
    for file in os.scandir('media/'):
        os.remove(file.path)

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance

            r = eyeoftruth(net)

            if r==1:
                context = {
                    'msg': '이 사진은 조작되었습니다',
                    'manipulated': 1,
                }
                visualize(net)

                return render(request, 'basecamp/analysis.html', context)
            else:
                context = {
                    'msg': '이 사진은 조작되지 않았습니다',
                    'manipulated': 0,
                }
                return render(request, 'basecamp/analysis.html', context)

    else:
        form = UploadFileForm()
    return render(request, 'blog/upload.html', {'form': form})


