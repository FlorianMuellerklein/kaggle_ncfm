import os
import glob
import json
import numpy as np

from skimage.io import imread

# to make the crop in PIL do the following
# im.crop('x', 'y', 'x'+'width', 'y'+'height')
# where x, y, width, height are values from json

# load the json file
with open('data/train_box.json') as data_file:
    fish_json = json.load(data_file)

# should be 1000 images in the test folder
print len(fish_json)

# loop through each record in the json and record file name and coords
fish_files = []
fish_coords = []
for i in range(len(fish_json)):
    # save the filename
    fish_files.append(fish_json[i]['filename'])

    print fish_json[i]['filename']
    # save the coords
    fish_coords.append([fish_json[i]['annotations'][0]['x'],
                        fish_json[i]['annotations'][0]['y'],
                        fish_json[i]['annotations'][0]['width'],
                        fish_json[i]['annotations'][0]['height']])

np_coords = np.asarray(fish_coords, dtype=np.float32)

np.save('data/cache/box_coords.npy', np_coords)

X_train = np.empty(shape=(len(fish_files), 3, 448, 448), dtype='float32')
print 'Read train images'
for i, fl in enumerate(fish_files):
    print fl
    img = imread('data/' +  fl)
    img = img.transpose(2, 0, 1).astype('float32')
    X_train[i] = img

print X_train.shape
np.save('data/cache/box_imgs.npy', X_train)
