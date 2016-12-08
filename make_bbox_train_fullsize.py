import os
import gzip
import glob
import json
import pickle
import numpy as np

from skimage.io import imread
from skimage import transform


fish_files = []
fish_coords = []
X_test = np.empty(shape=(1000, 3, 224, 224), dtype='float32')

img_counter = 0

path = os.path.join('data', 'test_stg1', '.jpg')

for fish_fn in path:
    print fish_fn
    img = imread(fish_fn)
    width = img.shape[1]
    height = img.shape[0]
    img = transform.resize(img, output_shape=(224, 224, 3), preserve_range=True)
    img = img.transpose(2,0,1).astype('float32')
    X_train[img_counter] = img
    img_counter += 1

    # save the coords, we are only taking the first fish in the annotations
    # scale them between 0 and 1 to make training behave smoother
    fish_coords.append([fish_json[i]['annotations'][0]['x'] / width,
                        fish_json[i]['annotations'][0]['y'] / height,
                        fish_json[i]['annotations'][0]['width'] / width,
                        fish_json[i]['annotations'][0]['height'] / height])

np_coords = np.asarray(fish_coords, dtype='float32')
np.save('data/cache/bbox_train_coords_fullsize.npy', np_coords)
np.save('data/cache/bbox_train_imgs_fullsize.npy', X_train)

# save file names
f = gzip.open('data/cache/file_names_train.pklz', 'wb')
pickle.dump(fish_files, f)
f.close()
