import time
import random
import numpy as np
import pandas as pd

import multiprocessing as mp

from matplotlib import pyplot
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

                # set empty copy to hold augmented images so that we don't overwrite
                X_batch_aug = np.empty(shape = (X_batch.shape[0], 3, 448, 448), dtype = 'float32')

                # random translations
                trans_1 = random.randint(-12,12)
                trans_2 = random.randint(-12,12)
                crop_amt = ((12 - trans_1, 12 + trans_1), (12 - trans_2, 12 + trans_2), (0,0))

                # set the transform parameters for skimage.transform.warp
                # have to shift to center and then shift back after transformation otherwise
                # rotations will make image go out of frame
                center_shift   = np.array((448, 448)) / 2. - 0.5
                tform_center   = transform.SimilarityTransform(translation=-center_shift)
                tform_uncenter = transform.SimilarityTransform(translation=center_shift)

                tform_aug = transform.AffineTransform(translation = (trans_1, trans_2))

                tform = tform_center + tform_aug + tform_uncenter

                r_intensity = random.randint(0,1)
                g_intensity = random.randint(0,1)
                b_intensity = random.randint(0,1)
                intensity_scaler = random.uniform(-0.15, 0.15)

                # flip left-right choice
                flip_lr = random.randint(0,1)

                # images in the batch do the augmentation
                for j in range(X_batch.shape[0]):
                    img = X_batch[j]
                    img = img.transpose(1, 2, 0)
                    img_aug = np.zeros((448, 448, 3))
                    for k in range(0,3):
                        img_aug[:, :, k] = fast_warp(img[:, :, k], tform, output_shape = (448, 448))

                    if r_intensity == 1:
                        img_aug[:, :, 0] *= intensity_scaler
                    if g_intensity == 1:
                        img_aug[:, :, 1] *= intensity_scaler
                    if b_intensity == 1:
                        img_aug[:, :, 2] *= intensity_scaler

                    # flip the image lr
                    if flip_lr:
                        img_aug = np.fliplr(img_aug)

                    X_batch_aug[j] = img_aug.transpose(2, 0, 1)

                    # do the same translations for the bounding box
                    y_batch[j][0] += (trans_1 / 448.)
                    y_batch[j][1] += (trans_2 / 448.)
                    y_batch[j][2] += (trans_1 / 448.)
                    y_batch[j][3] += (trans_2 / 448.)

                    # if we flip the images then we also need to flip the bounding box
                    if flip_lr:
                        y_batch[j][0] = 1. - y_batch[j][0] - y_batch[j][2]

                yield [X_batch_aug, y_batch]

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
