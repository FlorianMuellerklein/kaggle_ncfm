import os
import gzip
import glob
import json
import pickle
import numpy as np

from PIL import Image
from skimage.io import imread
from skimage import transform


# load the json file
with open('data/bbox_train_clean_fullsize.json') as data_file:
    fish_json = json.load(data_file)

# should be 1000 images in the test folder
print len(fish_json)

#fish_dict = {0:'ALB', 1:'BET', 2:'DOL', 3:'LAG', 4:'NoF', 5:'OTHER', 6:'SHARK', 7:'YFT'}

fish_type = []
fish_files = []
fish_coords = []
X_train = []

img_counter = 0

for i in range(len(fish_json)):
    fish_fn = fish_json[i]['filename']
    print fish_fn
    if os.path.exists(fish_fn):
        if 'ALB' in str(fish_fn):
            fish_type.append('ALB')
        if 'BET' in str(fish_fn):
            fish_type.append('BET')
        if 'DOL' in str(fish_fn):
            fish_type.append('DOL')
        if 'LAG' in str(fish_fn):
            fish_type.append('LAG')
        if 'NoF' in str(fish_fn):
            fish_type.append('NoF')
        if 'OTHER' in str(fish_fn):
            fish_type.append('OTHER')
        if 'SHARK' in str(fish_fn):
            fish_type.append('SHARK')
        if 'YFT' in str(fish_fn):
            fish_type.append('YFT')

        fish_files.append(fish_fn)
        img = imread(fish_fn)
        width = img.shape[1]
        height = img.shape[0]
        img = transform.resize(img, output_shape=(224, 224, 3), preserve_range=True)
        img = img.astype('float32')
        X_train.append(img)
        img_counter += 1

        # save the coords, we are only taking the first fish in the annotations
        # scale them between 0 and 1 to make training behave smoother
        fish_coords.append([fish_json[i]['annotations'][0]['x'] / width,
                            fish_json[i]['annotations'][0]['y'] / height,
                            fish_json[i]['annotations'][0]['width'] / width,
                            fish_json[i]['annotations'][0]['height'] / height])

        # crop the fish out of the training data so that we have perfect crops
        train_img = Image.open(fish_fn)
        side = max(fish_json[i]['annotations'][0]['width'], fish_json[i]['annotations'][0]['height'])
        im_cropped = train_img.crop((fish_json[i]['annotations'][0]['x'], fish_json[i]['annotations'][0]['y'], fish_json[i]['annotations'][0]['x']+side, fish_json[i]['annotations'][0]['y']+side))
        im_cropped.save(fish_fn.split('.jpg')[0] + '_cropped.jpg')

np_coords = np.asarray(fish_coords, dtype='float32')
fish_type = np.asarray(fish_type)
X_train = np.asarray(X_train, dtype='float32')
np.save('data/cache/bbox_train_clean_coords_fullsize.npy', np_coords)
np.save('data/cache/bbox_train_clean_labels.npy', fish_type)
np.save('data/cache/bbox_train_clean_imgs_fullsize.npy', X_train)

# save file names
f = gzip.open('data/cache/file_names_train_clean.pklz', 'wb')
pickle.dump(fish_files, f)
f.close()
