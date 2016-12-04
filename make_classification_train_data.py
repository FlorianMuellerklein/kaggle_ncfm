import os
import gzip
import glob
import json
import pickle
import numpy as np

from skimage.io import imread
from skimage import transform

folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

X_train = np.empty(shape=(3777,3,224,224), dtype='float32')
y_train = []
img_count = 0
for fish_type in folders:
    path = os.path.join('data', 'train', 'train', fish_type, '*_cropped.jpg')
    files = glob.glob(path)
    for fl in files:
        print fl, img_count
        img = imread(fl)
        img = transform.resize(img, output_shape=(224,224,3), preserve_range=True)
        img = img.transpose(2,0,1).astype('float32')
        X_train[img_count] = img
        y_train.append(fish_type)
        img_count += 1

print 'X_train shape:', X_train.shape
np.save('data/cache/X_train_classification.npy', X_train)
# save labels
f = gzip.open('data/cache/y_train_classification_labels.pklz', 'wb')
pickle.dump(y_train, f)
f.close()
