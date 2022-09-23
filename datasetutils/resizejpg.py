from PIL import Image
import os, sys
import glob

root_dir = "./dataset/alb/train/"


for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
    print(filename)
    im = Image.open(filename)
    imResize = im.resize((512,512), Image.ANTIALIAS)
    imResize.save(filename)