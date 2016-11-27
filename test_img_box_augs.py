import time
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imshow
from skimage.util import crop
from skimage import transform, filters, exposure

X_dat = np.load('data/cache/box_imgs.npy', mmap_mode='c')
y_dat = np.load('data/cache/box_coords.npy')

test_img = X_dat[12] / 255.
test_img = test_img.transpose(1,2,0)
test_coords = y_dat[12]

for i in range(10):
    img = np.copy(test_img)
    coords = np.copy(test_coords)
    print 'Orig:', coords
    # random translations
    trans_1 = random.randint(-100,100)
    trans_2 = random.randint(-100,100)

    # set the transform parameters for skimage.transform.warp
    # have to shift to center and then shift back after transformation otherwise
    # rotations will make image go out of frame
    #center_shift   = np.array((448, 448)) / 2. - 0.5
    #tform_center   = transform.SimilarityTransform(translation=-center_shift)
    #tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(translation=(trans_1, trans_2))

    #tform = tform_center + tform_aug + tform_uncenter

    r_intensity = random.randint(0,1)
    g_intensity = random.randint(0,1)
    b_intensity = random.randint(0,1)
    intensity_scaler = random.uniform(-0.15, 0.15)

    # flip left-right choice
    flip_lr = random.randint(0,1)

    img = transform.warp(img, tform_aug)

    # do the same translations for the bounding box
    coords[0] -= (trans_1)
    coords[1] -= (trans_2)

    if flip_lr:
        img = np.fliplr(img)
        #old_x = coords[0]
        #old_width = coords[2]
        coords[0] = img.shape[0] - coords[0] - coords[2]

    fig, ax = plt.subplots(1)

    ax.imshow(img)

    print 'New :', coords

    # draw the true bounding box in yellow
    rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3],
                                  linewidth=2, edgecolor='y',facecolor='none')
    # add the rectangles
    ax.add_patch(rect)

    # display
    plt.show()
