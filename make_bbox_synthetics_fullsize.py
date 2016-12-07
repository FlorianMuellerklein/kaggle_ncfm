import os
import gzip
import glob
import json
import pickle
import numpy as np

from skimage.io import imread
from skimage import transform


# load the json file
with open('data/bbox_synthetics_fullsize.json') as data_file:
    fish_json = json.load(data_file)

# should be 1000 images in the test folder
print len(fish_json)


fish_files = []
fish_coords = []
X_synth = np.empty(shape=(1000, 3, 224, 224), dtype='float32')

img_counter = 0

for i in range(len(fish_json)):
    fish_fn = fish_json[i]['filename']
    print fish_fn
    fish_files.append(fish_fn)
    img = imread(fish_fn)
    width = img.shape[1]
    height = img.shape[0]
    img = transform.resize(img, output_shape=(224, 224, 3), preserve_range=True)
    img = img.transpose(2,0,1).astype('float32')
    X_synth[i] = img
    img_counter += 1

    # save the coords, we are only taking the first fish in the annotations
    # scale them between 0 and 1 to make training behave smoother
    fish_coords.append([fish_json[i]['annotations'][0]['x'] / width,
                        fish_json[i]['annotations'][0]['y'] / height,
                        fish_json[i]['annotations'][0]['width'] / width,
                        fish_json[i]['annotations'][0]['height'] / height])

np_coords = np.asarray(fish_coords, dtype='float32')
np.save('data/cache/bbox_synthetics_coords_fullsize.npy', np_coords)
np.save('data/cache/bbox_synthetics_imgs_fullsize.npy', X_synth)

# save file names
f = gzip.open('data/cache/file_names_synthetics.pklz', 'wb')
pickle.dump(fish_files, f)
f.close()
