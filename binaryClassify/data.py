import tqdm
import torch
import numpy as np
import os
from PIL import Image

# load single, double dataset and q_table
def TrainData(data_path): # An abstract class representing a Dataset
    os.chdir(data_path)
    PIC_DIR = f'./faces/'

    images_COUNT = 100
    ORIG_WIDTH = 178
    ORIG_HEIGHT = 208
    diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2

    WIDTH = 128
    HEIGHT = 128

    crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)

    images = []
    for pic_file in tqdm(os.listdir(PIC_DIR)[:images_COUNT]):
        pic = Image.open(PIC_DIR + pic_file).crop(crop_rect)
        pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
        images.append(np.uint8(pic))

    images = np.array(images) / 255


