import gzip
import time
import h5py
import math
import pickle
import random
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.io import imread
from PIL import Image

from keras.optimizers import SGD, Adam
from keras.layers import merge, Input, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.utils.layer_utils import convert_all_kernels_in_model, print_summary

from sklearn.cross_validation import train_test_split

from utils import threaded_batch_iter_loc, bbox_tta

# set up training params
ITERS = 100
BATCHSIZE = 64
LR_SCHEDULE = {
     0: 0.0001,
    60: 0.00001,
}

'''
------------------------------------------------------------------------------------------------
Set up the ResNet-50 model.
Code from the pretrained models in keras repo
https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
Instead of using their default capabilities I changed some things for our problem.
------------------------------------------------------------------------------------------------
'''

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50():
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

    img_input = Input(shape=(3, 224, 224), name='Input', dtype='float32')

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # input images are 448,448 so we 2,2 maxpool stride 2 to downsample 2x
    #x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(img_input)

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dropout(p=0.75)(x)

    x = Dense(4, activation='sigmoid', name='out')(x)

    model = Model(img_input, x)

    # load weights
    #f = h5py.File('data/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')
    model.load_weights('data/resnet50_weights_th_dim_ordering_th_kernels_notop.h5', by_name=True)

    if K.backend() == 'theano':
        convert_all_kernels_in_model(model)

    return model

#print_summary(resnet.layers)

'''
------------------------------------------------------------------------------------------------
Load the images and split into  training and validation.
------------------------------------------------------------------------------------------------
'''

X_dat = np.load('data/cache/test_imgs_fullsize.npy')

# load file names
f = gzip.open('data/cache/file_names_test.pklz', 'rb')
filez = pickle.load(f)
f.close()


# Do preprocessing consistent with how it was done when the ImageNet images were
# used to originally train the model
# subtract channels means from ImageNet
X_dat[:, 0, :, :] -= 103.939
X_dat[:, 1, :, :] -= 116.779
X_dat[:, 2, :, :] -= 123.68
# 'RGB'->'BGR'
X_dat = X_dat[:, [2,1,0], :, :]

'''
------------------------------------------------------------------------------------------------
Compile and Train the Model

To try:
1) Felix Lau did 1000 iterations with early stopping and patience of 150 for the whale competition.
   He wasn't able to use pretrained models in that competition though.
   This one is initialized with ImageNet weights.
2) Mess with dropout rate at the end of the ResNet
3) Try dropout within the ResNet like how wide-resnets do it
4) Try a different localization architecture, ResNet might be too big for such a small dataset
------------------------------------------------------------------------------------------------
'''

resnet = ResNet50()

'''
------------------------------------------------------------------------------------------------
Test an image and see how it did
------------------------------------------------------------------------------------------------
'''


#resnet.load_weights('weights/best_resnet_loc_0.h5')
#guess_coords_1 = bbox_tta(resnet, X_dat)

resnet.load_weights('weights/best_resnet_synth_loc_1.h5')
guess_coords_2 = bbox_tta(resnet, X_dat)

resnet.load_weights('weights/best_resnet_synth_loc_2.h5')
guess_coords_3 = bbox_tta(resnet, X_dat)

resnet.load_weights('weights/best_resnet_synth_loc_3.h5')
guess_coords_4 = bbox_tta(resnet, X_dat)

resnet.load_weights('weights/best_resnet_synth_loc_4.h5')
guess_coords_5 = bbox_tta(resnet, X_dat)

guess_coords = (guess_coords_2 + guess_coords_3 +
                guess_coords_4 + guess_coords_5) / 4.0

# some things to try
# 1) inflate the bounding box by some amount, say 10%, to allow for some error
# 2) force bounding box to be square, by taking the max of width and height and setting both to that
# 3)


i = 0
for choice in range(X_dat.shape[0]):

    # display the test image and it's bounding box
    test_file = filez[choice]
    test_img = Image.open(test_file)

    # get the guess coordinates
    guess = guess_coords[choice]

    # put bounding box back into original size
    # PIL does width, height
    # skimage does rows (height), cols (width) so we need to swap indexes if we switch
    guess[0] *= test_img.size[0]
    guess[1] *= test_img.size[1]
    guess[2] *= test_img.size[0]
    guess[3] *= test_img.size[1]

    # inflate bounding box by 10%
    guess[0] -= 0.5 * (0.25 * guess[2])
    guess[1] -= 0.5 * (0.25 * guess[3])
    guess[2] += 0.5 * (0.25 * guess[2])
    guess[3] += 0.5 * (0.25 * guess[3])

    # force the crop box to be square so that we dont' mess with aspect ratio
    guess[2] = max(guess[2], guess[3])
    guess[3] = max(guess[2], guess[3])

    # crop the fish out of the image
    im_cropped = test_img.crop((guess[0], guess[1], guess[0]+guess[2], guess[1]+guess[3]))
    im_cropped.save(test_file.split('.jpg')[0] + '_cropped.jpg')
