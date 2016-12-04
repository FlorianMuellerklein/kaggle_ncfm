import os
import gzip
import glob
import json
import pickle
import numpy as np

from skimage.io import imread
from skimage import transform

y_test = []
X_test = np.empty(shape=(1000,3,224,224), dtype='float32')
img_count = 0
path = os.path.join('data', 'test_stg1', '*_cropped.jpg')
files = glob.glob(path)
for fl in files:
    print fl, img_count
    img = imread(fl)
    img = transform.resize(img, output_shape=(224,224,3), preserve_range=True)
    img = img.transpose(2,0,1).astype('float32')
    X_test[img_count] = img
    y_test.append(fl.split('/test_stg1/')[1])
    img_count += 1

for i in range(len(y_test)):
    y_test[i] = y_test[i].replace('_cropped', '')

print y_test[0]

print 'X_test shape:', X_test.shape
np.save('data/cache/X_test_classification.npy', X_test)
# save labels
f = gzip.open('data/cache/y_test_classification_labels.pklz', 'wb')
pickle.dump(y_test, f)
f.close()
