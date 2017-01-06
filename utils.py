import time
import random
import numpy as np
import pandas as pd

from PIL import Image

import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.utils import shuffle

from skimage.io import imshow
from skimage.util import crop
from skimage import transform, filters, exposure

'''
This file contains the real-time batch iterators for training the various networks,
as well as the test-time augmentations for making predictions.

The batch iterators are multithreaded so that the GPU will not have to wait for the
image augmentations to be complete before it can start processing the batch. The augmentations
are done in the background while the GPU runs the previous batch.

The test time augmentations are for creating various versions of the same images for each
prediction then we average those predictions for the final one. For example we'll take five
translations of a single image at random, make a prediction for each, then average all of those.
'''

# fast_warp is faster than transform.warp
def fast_warp(img, tf, output_shape, mode='reflect'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

# create a function to randomly move fish around the image
def fish_mover(fish_img, class_lbl, coords):
    # check if the class is NoF
    if class_lbl != 'NoF':
        # convert the image to PIL so that we can overlay
        img = Image.fromarray(np.uint8(fish_img))

        # get the label
        lbl = class_lbl

        # get the coords
        xywh = coords * 224.
        x_orig = xywh[0]
        y_orig = xywh[1]
        width = xywh[2]
        height = xywh[3]

        # crop out the fish and big area around it
        fish = img.crop((x_orig-width, y_orig-height, x_orig + 2.*width, y_orig + 2*height))

        # make a random crop from the image to replace with the fish crop
        # take the same size crop directly above or below the fish
        if y_orig > 112:
            y_rnd = y_orig - height
        else:
            y_rnd = y_orig + 2*height

        # crop out the fill region
        fill = img.crop((x_orig, y_rnd, x_orig+width, y_rnd+height))

        # get random coords to place fish back
        x = random.randint(0,168)
        y = random.randint(0,168)

        # replace the values in coords
        new_coords = np.copy(coords)
        new_coords[0] = x
        new_coords[1] = y
        new_coords[2] = width
        new_coords[3] = height

        # if the fish goes off of the screen set the bbox to full img size
        # make sure to set the label to NoF if this happens
        if new_coords[0] > 224.:
            lbl = 'NoF'
            new_coords[0] = 0.
            new_coords[1] = 0.
            new_coords[2] = 1.
            new_coords[3] = 1.

        if new_coords[1] > 224.:
            lbl = 'NoF'
            new_coords[0] = 0.
            new_coords[1] = 0.
            new_coords[2] = 1.
            new_coords[3] = 1.

        new_coords /= 224.

        # paste the fill into the fish location
        img.paste(fill, (int(x_orig), int(y_orig)))

        # paste the fish into the random location
        img.paste(fish, (int(x - width) , int(y - height)))

        # turn the PIL image into numpy array
        img_np = np.asarray(img, dtype='float32')
    else:
        img_np = fish_img
        lbl = class_lbl
        new_coords = coords

    # return the moved fish img, new label, and new coords
    return img_np, lbl, new_coords

'''
--------------------------------------------------------------------------------------------------
Batch iterator for classification
--------------------------------------------------------------------------------------------------
'''

class threaded_batch_iter_class(object):
    '''
    Batch iterator to make transformations on the data
    '''
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __call__(self, X, y):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        '''
        q = mp.Queue(maxsize=32)

        def _gen_batches():
            num_samples = len(self.X)
            idx = np.random.permutation(num_samples)
            batches = range(0, num_samples - self.batchsize + 1, self.batchsize)
            for batch in batches:
                X_batch = self.X[idx[batch:batch + self.batchsize]]
                y_batch = self.y[idx[batch:batch + self.batchsize]]

                # set copy to hold augmented images so that we don't overwrite
                X_batch_aug = np.copy(X_batch)
                y_batch_aug = np.copy(y_batch)

                # random translations
                trans_1 = random.randint(-20,20)
                trans_2 = random.randint(-20,20)

                # random zooms
                zoom = random.uniform(0.8, 1.2)

                # shearing
                shear_deg = random.uniform(-10,10)

                # rotate
                dorotate = random.randint(-45, 45)

                # set the transform parameters for skimage.transform.warp
                # have to shift to center and then shift back after transformation otherwise
                # rotations will make image go out of frame
                center_shift   = np.array((224, 224)) / 2. - 0.5
                tform_center   = transform.SimilarityTransform(translation=-center_shift)
                tform_uncenter = transform.SimilarityTransform(translation=center_shift)

                tform_aug = transform.AffineTransform(shear = np.deg2rad(shear_deg),
                                                      rotation = np.deg2rad(dorotate),
                                                      scale=(1/zoom, 1/zoom),
                                                      translation=(trans_1, trans_2))

                tform = tform_center + tform_aug + tform_uncenter

                # flip left-right choice
                flip_lr = random.randint(0,1)

                # flip up-down choice
                flip_ud = random.randint(0,1)

                # color intensity augmentation
                r_intensity = random.randint(0,1)
                g_intensity = random.randint(0,1)
                b_intensity = random.randint(0,1)
                intensity_scaler = random.randint(-10, 10)

                # images in the batch do the augmentation
                for j in range(X_batch_aug.shape[0]):
                    img = X_batch_aug[j]
                    #img = img.transpose(1, 2, 0)
                    img_aug = np.zeros((224, 224, 3))
                    for k in range(0,3):
                        img_aug[:,:,k] = fast_warp(img[:,:,k], tform, output_shape = (224, 224))

                    # flip the image lr
                    if flip_lr:
                        img_aug = np.fliplr(img_aug)

                    # flip up down if that is chosen
                    if flip_ud:
                        img_aug = np.flipud(img_aug)

                    if r_intensity == 1:
                        img_aug[:,:,0] += intensity_scaler
                    if g_intensity == 1:
                        img_aug[:,:,1] += intensity_scaler
                    if b_intensity == 1:
                        img_aug[:,:,2] += intensity_scaler

                    X_batch_aug[j] = img_aug

                    '''
                    # for debugging, display img and bounding box for each image in a batch
                    fig, ax = plt.subplots(1)
                    disp_img = img_aug
                    disp_img[:, :, 0] += 103.939
                    disp_img[:, :, 1] += 116.779
                    disp_img[:, :, 2] += 123.68
                    disp_img = disp_img[:,:,[2,1,0]]
                    ax.imshow(disp_img / 255.)
                    # display
                    plt.show()
                    print self.lblr.inverse_transform(y_batch_aug[j])
                    '''

                yield [X_batch_aug, y_batch_aug]

        def _producer(_gen_batches):
            batch_gen = _gen_batches()
            for data in batch_gen:
                q.put(data, block=True)
            q.put(None)
            q.close()

        thread = mp.Process(target=_producer, args=[_gen_batches])
        thread.daemon = True
        thread.start()

        for data in iter(q.get, None):
            yield data[0], data[1]

'''
--------------------------------------------------------------------------------------------------
Batch iterator for classification with synthetic images
--------------------------------------------------------------------------------------------------
'''

class threaded_batch_iter_class_synth(object):
    '''
    Batch iterator to make transformations on the data
    '''
    def __init__(self, batchsize, lblr):
        self.batchsize, self.lblr = batchsize, lblr

    def __call__(self, X, y, coords):
        self.X, self.y, self.coords = X, y, coords
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        '''
        q = mp.Queue(maxsize=32)

        def _gen_batches():
            num_samples = len(self.X)
            idx = np.random.permutation(num_samples)
            batches = range(0, num_samples - self.batchsize + 1, self.batchsize)
            for batch in batches:
                X_batch = self.X[idx[batch:batch + self.batchsize]]
                y_batch = self.y[idx[batch:batch + self.batchsize]]
                y_coords = self.coords[idx[batch:batch + self.batchsize]]

                # set copy to hold augmented images so that we don't overwrite
                X_batch_aug = np.copy(X_batch)
                y_batch_aug = np.copy(y_batch)
                y_batch_coords = np.copy(y_coords)

                # move fish choice
                mv_fish = random.randint(0,1)

                # random translations
                trans_1 = random.randint(-20,20)
                trans_2 = random.randint(-20,20)

                # random zooms
                zoom = random.uniform(0.8, 1.2)

                # shearing
                shear_deg = random.uniform(-10,10)

                # rotate
                dorotate = random.randint(-45, 45)

                # set the transform parameters for skimage.transform.warp
                # have to shift to center and then shift back after transformation otherwise
                # rotations will make image go out of frame
                center_shift   = np.array((224, 224)) / 2. - 0.5
                tform_center   = transform.SimilarityTransform(translation=-center_shift)
                tform_uncenter = transform.SimilarityTransform(translation=center_shift)

                tform_aug = transform.AffineTransform(shear = np.deg2rad(shear_deg),
                                                      rotation = np.deg2rad(dorotate),
                                                      scale=(1/zoom, 1/zoom),
                                                      translation=(trans_1, trans_2))

                tform = tform_center + tform_aug + tform_uncenter

                # flip left-right choice
                flip_lr = random.randint(0,1)

                # flip up-down choice
                flip_ud = random.randint(0,1)

                # color intensity augmentation
                r_intensity = random.randint(0,1)
                g_intensity = random.randint(0,1)
                b_intensity = random.randint(0,1)
                intensity_scaler = random.randint(-10, 10)

                # images in the batch do the augmentation
                for j in range(X_batch_aug.shape[0]):
                    # make a copy of the image to augment
                    img = X_batch_aug[j]

                    # if we choose to move the fish
                    if mv_fish:
                        img, lbl, new_coords = fish_mover(img, y_batch_aug[j], y_batch_coords[j])
                        y_batch_aug[j] = lbl

                    img_aug = np.zeros((224, 224, 3))
                    for k in range(0,3):
                        img_aug[:,:,k] = fast_warp(img[:,:,k], tform, output_shape = (224, 224))

                    # flip the image lr
                    if flip_lr:
                        img_aug = np.fliplr(img_aug)

                    # flip up down if that is chosen
                    if flip_ud:
                        img_aug = np.flipud(img_aug)

                    if r_intensity == 1:
                        img_aug[:,:,0] += intensity_scaler
                    if g_intensity == 1:
                        img_aug[:,:,1] += intensity_scaler
                    if b_intensity == 1:
                        img_aug[:,:,2] += intensity_scaler

                    X_batch_aug[j] = img_aug

                    '''
                    # for debugging, display img and bounding box for each image in a batch
                    print y_batch_aug[j]
                    fig, ax = plt.subplots(1)
                    disp_img = img_aug
                    ax.imshow(disp_img / 255.)
                    # display
                    plt.show()
                    '''

                # Do preprocessing consistent with how it was done when the ImageNet images were
                # used to originally train the model
                # 'RGB'->'BGR'
                X_batch_aug = X_batch_aug[:, :, :, [2,1,0]]
                # subtract channels means from ImageNet
                X_batch_aug[:, :, :, 0] -= 103.939
                X_batch_aug[:, :, :, 1] -= 116.779
                X_batch_aug[:, :, :, 2] -= 123.68

                yield [X_batch_aug, self.lblr.transform(y_batch_aug)]

        def _producer(_gen_batches):
            batch_gen = _gen_batches()
            for data in batch_gen:
                q.put(data, block=True)
            q.put(None)
            q.close()

        thread = mp.Process(target=_producer, args=[_gen_batches])
        thread.daemon = True
        thread.start()

        for data in iter(q.get, None):
            yield data[0], data[1]

'''
--------------------------------------------------------------------------------------------------
Batch iterator for localization
--------------------------------------------------------------------------------------------------
'''

class threaded_batch_iter_loc(object):
    '''
    Batch iterator to make transformations on the data
    '''
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __call__(self, X, y, coords):
        self.X, self.y, self.coords = X, y, coords
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        '''
        q = mp.Queue(maxsize=32)

        def _gen_batches():
            num_samples = len(self.X)
            idx = np.random.permutation(num_samples)
            batches = range(0, num_samples - self.batchsize + 1, self.batchsize)
            for batch in batches:
                X_batch = self.X[idx[batch:batch + self.batchsize]]
                y_batch = self.y[idx[batch:batch + self.batchsize]]
                y_coords = self.coords[idx[batch:batch + self.batchsize]]

                # set copy to hold augmented images so that we don't overwrite
                X_batch_aug = np.copy(X_batch)
                y_batch_aug = np.copy(y_batch)
                y_batch_coords = np.copy(y_coords)

                # move fish choice
                mv_fish = random.randint(0,1)

                # random translations
                trans_1 = random.randint(-25,25)
                trans_2 = random.randint(-25,25)

                tform_aug = transform.AffineTransform(translation=(trans_1, trans_2))

                # flip left-right choice
                flip_lr = random.randint(0,1)

                # flip up-down choice
                flip_ud = random.randint(0,1)

                # color intensity augmentation
                r_intensity = random.randint(0,1)
                g_intensity = random.randint(0,1)
                b_intensity = random.randint(0,1)
                intensity_scaler = random.randint(-20, 20)

                # images in the batch do the augmentation
                for j in range(X_batch_aug.shape[0]):
                    # make a copy of the image to augment
                    img = X_batch_aug[j]

                    # if we choose to move the fish
                    if mv_fish:
                        img, lbl, new_coords = fish_mover(img, y_batch_aug[j], y_batch_coords[j])
                        y_batch_coords[j] = new_coords

                    img_aug = np.zeros((224, 224, 3))
                    for k in range(0,3):
                        img_aug[:,:,k] = fast_warp(img[:,:,k], tform_aug, output_shape = (224, 224))

                    # do the same translations for the bounding box
                    y_batch_coords[j][0] -= (trans_1 / 224.)
                    y_batch_coords[j][1] -= (trans_2 / 224.)

                    # flip the image lr
                    if flip_lr:
                        img_aug = np.fliplr(img_aug)
                        y_batch_coords[j][0] = 1. - y_batch_coords[j][0] - y_batch_coords[j][2]

                    # flip up down if that is chosen
                    if flip_ud:
                        img_aug = np.flipud(img_aug)
                        y_batch_coords[j][1] = 1. - y_batch_coords[j][1] - y_batch_coords[j][3]

                    if r_intensity == 1:
                        img_aug[:,:,0] += intensity_scaler
                    if g_intensity == 1:
                        img_aug[:,:,1] += intensity_scaler
                    if b_intensity == 1:
                        img_aug[:,:,2] += intensity_scaler

                    '''
                    # for debugging, display img and bounding box for each image in a batch
                    fig, ax = plt.subplots(1)
                    disp_img = img_aug
                    ax.imshow(disp_img / 255.)
                    # draw the true bounding box in yellow
                    rect = patches.Rectangle((y_batch_coords[j][0] * 224., y_batch_coords[j][1] * 224.), y_batch_coords[j][2] * 224., y_batch_coords[j][3] * 224.,
                                              linewidth=2, edgecolor='y',facecolor='none')
                    # add the rectangles
                    ax.add_patch(rect)
                    # display
                    plt.show()
                    '''

                    X_batch_aug[j] = img_aug

                # Do preprocessing consistent with how it was done when the ImageNet images were
                # used to originally train the model
                # 'RGB'->'BGR'
                X_batch_aug = X_batch_aug[:, :, :, [2,1,0]]
                # subtract channels means from ImageNet
                X_batch_aug[:, :, :, 0] -= 103.939
                X_batch_aug[:, :, :, 1] -= 116.779
                X_batch_aug[:, :, :, 2] -= 123.68

                yield [X_batch_aug, y_batch_coords]

        def _producer(_gen_batches):
            batch_gen = _gen_batches()
            for data in batch_gen:
                q.put(data, block=True)
            q.put(None)
            q.close()

        thread = mp.Process(target=_producer, args=[_gen_batches])
        thread.daemon = True
        thread.start()

        for data in iter(q.get, None):
            yield data[0], data[1]

'''
--------------------------------------------------------------------------------------------------
Batch iterator for localization  and classification
--------------------------------------------------------------------------------------------------
'''

class threaded_batch_iter_loc_class(object):
    '''
    Batch iterator to make transformations on the data
    '''
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __call__(self, X, y_bb, y_class):
        self.X, self.y_bb, self.y_class = X, y_bb, y_class
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        '''
        q = mp.Queue(maxsize=32)

        def _gen_batches():
            num_samples = len(self.X)
            idx = np.random.permutation(num_samples)
            batches = range(0, num_samples - self.batchsize + 1, self.batchsize)
            for batch in batches:
                X_batch       = self.X[idx[batch:batch + self.batchsize]]
                y_bb_batch    = self.y_bb[idx[batch:batch + self.batchsize]]
                y_class_batch = self.y_class[idx[batch:batch + self.batchsize]]

                # set copy to hold augmented images so that we don't overwrite
                X_batch_aug = np.copy(X_batch)
                y_bb_batch_aug = np.copy(y_bb_batch)
                y_class_batch_aug = np.copy(y_class_batch)

                # random translations
                trans_1 = random.randint(-25,25)
                trans_2 = random.randint(-25,25)

                tform_aug = transform.AffineTransform(translation=(trans_1, trans_2))

                # flip left-right choice
                flip_lr = random.randint(0,1)

                # flip up-down choice
                flip_ud = random.randint(0,1)

                # color intensity augmentation
                r_intensity = random.randint(0,1)
                g_intensity = random.randint(0,1)
                b_intensity = random.randint(0,1)
                intensity_scaler = random.randint(-20, 20)

                # images in the batch do the augmentation
                for j in range(X_batch_aug.shape[0]):
                    img = X_batch_aug[j]
                    #img = img.transpose(1, 2, 0)
                    img_aug = np.zeros((224, 224, 3))
                    for k in range(0,3):
                        img_aug[:,:,k] = fast_warp(img[:,:,k], tform_aug, output_shape = (224, 224))

                    # do the same translations for the bounding box
                    y_bb_batch_aug[j][0] -= (trans_1 / 224.)
                    y_bb_batch_aug[j][1] -= (trans_2 / 224.)

                    # flip the image lr
                    if flip_lr:
                        img_aug = np.fliplr(img_aug)
                        y_bb_batch_aug[j][0] = 1. - y_bb_batch_aug[j][0] - y_bb_batch_aug[j][2]

                    # flip up down if that is chosen
                    if flip_ud:
                        img_aug = np.flipud(img_aug)
                        y_bb_batch_aug[j][1] = 1. - y_bb_batch_aug[j][1] - y_bb_batch_aug[j][3]

                    if r_intensity == 1:
                        img_aug[:,:,0] += intensity_scaler
                    if g_intensity == 1:
                        img_aug[:,:,1] += intensity_scaler
                    if b_intensity == 1:
                        img_aug[:,:,2] += intensity_scaler

                    '''
                    # for debugging, display img and bounding box for each image in a batch
                    fig, ax = plt.subplots(1)
                    disp_img = img_aug
                    disp_img[:, :, 0] += 103.939
                    disp_img[:, :, 1] += 116.779
                    disp_img[:, :, 2] += 123.68
                    disp_img = disp_img[:,:,[2,1,0]]
                    ax.imshow(disp_img / 255.)
                    # draw the true bounding box in yellow
                    rect = patches.Rectangle((y_bb_batch_aug[j][0] * 224., y_bb_batch_aug[j][1] * 224.), y_bb_batch_aug[j][2] * 224., y_bb_batch_aug[j][3] * 224.,
                                              linewidth=2, edgecolor='y',facecolor='none')
                    # add the rectangles
                    ax.add_patch(rect)
                    # display
                    plt.show()
                    '''

                    X_batch_aug[j] = img_aug

                yield [X_batch_aug, y_bb_batch_aug, y_class_batch_aug]

        def _producer(_gen_batches):
            batch_gen = _gen_batches()
            for data in batch_gen:
                q.put(data, block=True)
            q.put(None)
            q.close()

        thread = mp.Process(target=_producer, args=[_gen_batches])
        thread.daemon = True
        thread.start()

        for data in iter(q.get, None):
            yield data[0], data[1], data[2]


'''
--------------------------------------------------------------------------------------------------
Batch iterator for pseudo labels
--------------------------------------------------------------------------------------------------
'''

class threaded_batch_iter_pseudo(object):
    '''
    Batch iterator to make transformations on the data
    '''
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __call__(self, X, y, Xp, yp):
        self.X, self.y, self.Xp, self.yp = X, y, Xp, yp
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        '''
        q = mp.Queue(maxsize=32)

        def _gen_batches():
            BATCHSIZE = self.batchsize
            # calculate how big the batch from regular and pseudo data should be
            pBATCHSIZE = int(round(self.batchsize * 0.33))
            BATCHSIZE -= pBATCHSIZE
            # count number of samples
            num_samples = len(self.X)
            # do random permutation
            idx = np.random.permutation(num_samples)
            pidx = np.random.permutation(len(self.Xp))
            # loop over batches
            for i in range((num_samples + BATCHSIZE - 1) // BATCHSIZE):
                sl = slice(i*BATCHSIZE, (i+1) * BATCHSIZE)
                psl = slice(i*pBATCHSIZE, (i+1) * pBATCHSIZE)

                # set copy to hold augmented images so that we don't overwrite
                X_batch_aug = np.vstack((self.X[idx[sl]], self.Xp[pidx[psl]]))
                y_batch_aug = np.vstack((self.y[idx[sl]], self.yp[pidx[psl]]))

                X_batch_aug, y_batch_aug = shuffle(X_batch_aug, y_batch_aug)

                # random translations
                trans_1 = random.randint(-50,50)
                trans_2 = random.randint(-50,50)

                # random zooms
                zoom = random.uniform(0.8, 1.2)

                # shearing
                shear_deg = random.uniform(-5,5)

                tform_aug = transform.AffineTransform(shear = np.deg2rad(shear_deg),
                                                      scale=(1/zoom, 1/zoom),
                                                      translation=(trans_1, trans_2))

                # flip left-right choice
                flip_lr = random.randint(0,1)

                # flip up-down choice
                flip_ud = random.randint(0,1)

                # color intensity augmentation
                r_intensity = random.randint(0,1)
                g_intensity = random.randint(0,1)
                b_intensity = random.randint(0,1)
                intensity_scaler = random.randint(-20, 20)

                # images in the batch do the augmentation
                for j in range(X_batch_aug.shape[0]):
                    img = X_batch_aug[j]
                    img_aug = np.zeros((224, 224, 3))
                    for k in range(0,3):
                        img_aug[:,:,k] = fast_warp(img[:,:,k], tform_aug, output_shape = (224, 224))

                    # flip the image lr
                    if flip_lr:
                        img_aug = np.fliplr(img_aug)

                    # flip up down if that is chosen
                    if flip_ud:
                        img_aug = np.flipud(img_aug)

                    if r_intensity == 1:
                        img_aug[:,:,0] += intensity_scaler
                    if g_intensity == 1:
                        img_aug[:,:,1] += intensity_scaler
                    if b_intensity == 1:
                        img_aug[:,:,2] += intensity_scaler

                    X_batch_aug[j] = img_aug

                yield [X_batch_aug, y_batch_aug]

        def _producer(_gen_batches):
            batch_gen = _gen_batches()
            for data in batch_gen:
                q.put(data, block=True)
            q.put(None)
            q.close()

        thread = mp.Process(target=_producer, args=[_gen_batches])
        thread.daemon = True
        thread.start()

        for data in iter(q.get, None):
            yield data[0], data[1]

'''
--------------------------------------------------------------------------------------------------
Test time augmentations
--------------------------------------------------------------------------------------------------
'''

def bbox_tta(model, data, num_tta=20):
    ''' TTA for bounding boxes'''
    # make a list to hold each prediction
    tta_preds = []

    # loop as many times as we designate for TTA
    for i in range(num_tta):
        print 'Running TTA ' + str(i+1) + ' ...'
        # copy the data so that we don't keep overwriting it with augs
        data_copy = np.copy(data)

        # random translations
        trans_1 = random.randint(-15,15)
        trans_2 = random.randint(-15,15)

        tform_aug = transform.AffineTransform(translation=(trans_1, trans_2))

        for j in range(data_copy.shape[0]):
            img = data_copy[j]
            #img = img.transpose(1,2,0)
            img_aug = np.zeros((224, 224, 3))

            for k in range(0,3):
                img_aug[:,:,k] = fast_warp(img[:,:,k], tform_aug, output_shape = (224, 224))

            data_copy[j] = img_aug

        # make predictions
        y_pred = model.predict(data_copy)

        # set the predictions back to what their original position would be
        y_pred[:,0] += (trans_1 / 224.)
        y_pred[:,1] += (trans_2 / 224.)

        # append prediction to list
        tta_preds.append(y_pred)

    # take the average of each of the TTA results
    #for j in range(len(tta_preds)):
    #    if j == 0:
    #        preds_out = tta_preds[j]
    #    else:
    #        preds_out += tta_preds[j]

    #preds_out /= float(len(tta_preds))

    preds_out = np.zeros((len(data),4))
    for c in range(len(preds_out)):
        # get min x coord
        x_coord = []
        for x in range(len(tta_preds)):
            x_coord.append(tta_preds[x][c][0])
        preds_out[c][0] = min(x_coord)
        # get min y coord
        y_coord = []
        for y in range(len(tta_preds)):
            y_coord.append(tta_preds[y][c][1])
        preds_out[c][1] = min(y_coord)
        # get max width
        w_val = []
        for w in range(len(tta_preds)):
            w_val.append(tta_preds[w][c][2])
        preds_out[c][2] = max(w_val)
        # get max height
        h_val = []
        for h in range(len(tta_preds)):
            h_val.append(tta_preds[h][c][3])
        preds_out[c][3] = max(h_val)

    # take the average of each of the TTA results
    #avg_preds = (tta_preds[0] + tta_preds[1] + tta_preds[2] + tta_preds[3] + tta_preds[4] +
    #             tta_preds[5] + tta_preds[6] + tta_preds[7] + tta_preds[8] + tta_preds[9]) / 10.

    return preds_out

def class_tta(model, data, num_tta=10, model_num=0):
    '''TTA for classification'''
    # make a list to hold each prediction
    tta_preds = []

    # loop as many times as we designate for TTA
    for i in range(num_tta):
        print 'Running model' + str(model_num) + ' TTA ' + str(i+1) + ' ...'
        # copy the data so that we don't keep overwriting it with augs
        data_copy = np.copy(data)

        # random translations
        trans_1 = random.randint(-15,15)
        trans_2 = random.randint(-15,15)

        tform_aug = transform.AffineTransform(translation=(trans_1, trans_2))

        for j in range(data_copy.shape[0]):
            img = data_copy[j]
            #img = img.transpose(1,2,0)
            img_aug = np.zeros((224, 224, 3))

            for k in range(0,3):
                img_aug[:,:,k] = fast_warp(img[:,:,k], tform_aug, output_shape = (224, 224))

            data_copy[j] = img_aug #.transpose(2, 0, 1)

        # make predictions
        y_pred = model.predict(data_copy)

        # append prediction to list
        tta_preds.append(y_pred)


    # take the average of each of the TTA results
    for j in range(len(tta_preds)):
        if j == 0:
            avg_preds = tta_preds[j]
        else:
            avg_preds += tta_preds[j]

    avg_preds /= float(len(tta_preds))

    # take the average of each of the TTA results
    #avg_preds = (tta_preds[0] + tta_preds[1] + tta_preds[2] + tta_preds[3] + tta_preds[4] +
    #             tta_preds[5] + tta_preds[6] + tta_preds[7] + tta_preds[8] + tta_preds[9]) / 10.

    return avg_preds
