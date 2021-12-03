import os
from PIL import Image

dir = '/fake_img'
new_dir = '../pythonProject1/binaryClassify/resized_fake_img'
file_list = os.listdir(dir)

for i in range(len(file_list)):
    file_name = dir + '\\' + file_list[i]
    img = Image.open(file_name)
    img_resized = img.convert("RGB").resize((256,256))
    img_resized.save(new_dir + '\\' + file_list[i])


