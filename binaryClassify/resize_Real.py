import os
from PIL import Image

dir = '/real_img'
new_dir = 'C:\\Users\\owner\\PycharmProjects\\pythonProject\\resized_realimg'
file_list = os.listdir(dir)




for i in file_list:
    img = Image.open(dir+'\\'+i)
    area = (110, 100, 800, 790)
    cropped_img =img.crop(area)
    img_resized = cropped_img.convert("RGB").resize((256, 256))
    img_resized.save(new_dir + '\\' + i)


