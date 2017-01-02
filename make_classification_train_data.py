import os
import gzip
import glob
import json
import pickle
import numpy as np

from skimage.io import imread
from skimage import transform

folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

X_train = []
y_train = []
img_count = 0
for fish_type in folders:
    path = os.path.join('data', 'train_clean', fish_type, '*_cropped.jpg')
    files = glob.glob(path)
    for fl in files:
        print fl, img_count
        img = imread(fl)
        img = transform.resize(img, output_shape=(224,224,3), preserve_range=True)
        #img = img.transpose(2,0,1).astype('float32')
        X_train.append(img.astype('float32'))
        y_train.append(fish_type)
        img_count += 1

X_train = np.asarray(X_train, dtype='float32')
print 'X_train shape:', X_train.shape
np.save('data/cache/X_train_classification_clean.npy', X_train)
# save labels
f = gzip.open('data/cache/y_train_classification_labels_clean.pklz', 'wb')
pickle.dump(y_train, f)
f.close()
