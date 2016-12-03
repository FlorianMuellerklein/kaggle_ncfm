import time
import random
import numpy as np
import pandas as pd

import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.io import imshow
from skimage.util import crop
from skimage import transform, filters, exposure


def fast_warp(img, tf, output_shape, mode='reflect'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

class threaded_batch_iter_loc(object):
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
                trans_1 = random.randint(-50,50)
                trans_2 = random.randint(-50,50)

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
                    img = img.transpose(1, 2, 0)
                    img_aug = np.zeros((224, 224, 3))
                    for k in range(0,3):
                        img_aug[:,:,k] = fast_warp(img[:,:,k], tform_aug, output_shape = (224, 224))

                    # do the same translations for the bounding box
                    y_batch_aug[j][0] -= (trans_1 / 224.)
                    y_batch_aug[j][1] -= (trans_2 / 224.)

                    # flip the image lr
                    if flip_lr:
                        img_aug = np.fliplr(img_aug)
                        y_batch_aug[j][0] = 1. - y_batch_aug[j][0] - y_batch_aug[j][2]

                    # flip up down if that is chosen
                    if flip_ud:
                        img_aug = np.flipud(img_aug)
                        y_batch_aug[j][1] = 1. - y_batch_aug[j][1] - y_batch_aug[j][3]

                    if r_intensity == 1:
                        img_aug[:,:,0] += intensity_scaler
                    if g_intensity == 1:
                        img_aug[:,:,1] += intensity_scaler
                    if b_intensity == 1:
                        img_aug[:,:,2] += intensity_scaler

                    '''
                    # for debugging, display img and bounding box for each image in a batch
                    fig, ax = plt.subplots(1)
                    disp_img = img_aug[:,:,[2,1,0]]
                    disp_img[:, :, 0] += 103.939
                    disp_img[:, :, 1] += 116.779
                    disp_img[:, :, 2] += 123.68
                    ax.imshow(disp_img / 255.)
                    # draw the true bounding box in yellow
                    rect = patches.Rectangle((y_batch_aug[j][0] * 224., y_batch_aug[j][1] * 224.), y_batch_aug[j][2] * 224., y_batch_aug[j][3] * 224.,
                                              linewidth=2, edgecolor='y',facecolor='none')
                    # add the rectangles
                    ax.add_patch(rect)
                    # display
                    plt.show()
                    '''

                    X_batch_aug[j] = img_aug.transpose(2, 0, 1)

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

def bbox_tta(model, data):
    # make a list to hold each prediction
    tta_preds = []

    # loop as many times as we designate for TTA
    for i in range(5):
        print 'Running TTA ' + str(i+1) + ' ...'
        # copy the data so that we don't keep overwriting it with augs
        data_copy = np.copy(data)

        # random translations
        trans_1 = random.randint(-15,15)
        trans_2 = random.randint(-15,15)

        tform_aug = transform.AffineTransform(translation=(trans_1, trans_2))

        for j in range(data_copy.shape[0]):
            img = data_copy[j]
            img = img.transpose(1,2,0)
            img_aug = np.zeros((224, 224, 3))

            for k in range(0,3):
                img_aug[:,:,k] = fast_warp(img[:,:,k], tform_aug, output_shape = (224, 224))

            data_copy[j] = img_aug.transpose(2, 0, 1)

        # make predictions
        y_pred = model.predict(data_copy)

        print y_pred.shape

        # set the predictions back to what their original position would be
        #y_pred[:,0] += (trans_1 / 224.)
        #y_pred[:,1] += (trans_2 / 224.)

        # append prediction to list
        tta_preds.append(y_pred)

    # take the average of each of the TTA results
    avg_preds = (tta_preds[0] + tta_preds[1] + tta_preds[2] + tta_preds[3] + tta_preds[4]) / 5.

    return avg_preds
