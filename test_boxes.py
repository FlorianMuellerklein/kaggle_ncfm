import time
import h5py
import math
import random
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

from utils import threaded_batch_iter_loc

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

    img_input = Input(shape=(3, 448, 448), name='Input', dtype='float32')

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # input images are 448,448 so we 2,2 maxpool stride 2 to downsample 2x
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(img_input)

    x = ZeroPadding2D((3, 3))(x)
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

'''
------------------------------------------------------------------------------------------------
Load the images and split into  training and validation.
------------------------------------------------------------------------------------------------
'''

X_dat = np.load('data/cache/box_imgs.npy', mmap_mode='c')
y_dat = np.load('data/cache/box_coords.npy')
y_dat /= 448.

idx = np.random.permutation(len(X_dat))

X_train = X_dat[idx[:3000]]
y_train = y_dat[idx[:3000]]
X_test = X_dat[idx[3000:]]
y_test = y_dat[idx[3000:]]

print 'Training Shape:', X_train.shape, y_train.shape
print 'Validation Shape:', X_test.shape, y_test.shape

# Do preprocessing consistent with how it was done when the ImageNet images were
# used to originally train the model
# subtract channels means from ImageNet
X_train[:, 0, :, :] -= 103.939
X_train[:, 1, :, :] -= 116.779
X_train[:, 2, :, :] -= 123.68
# 'RGB'->'BGR'
X_train = X_train[:, [2,1,0], :, :]

X_test[:, 0, :, :] -= 103.939
X_test[:, 1, :, :] -= 116.779
X_test[:, 2, :, :] -= 123.68
# 'RGB'->'BGR'
X_test = X_test[:, [2,1,0], :, :]


resnet = ResNet50()

'''
------------------------------------------------------------------------------------------------
Test an image and see how it did
------------------------------------------------------------------------------------------------
'''


resnet.load_weights('weights/best_resnet_loc_0.h5')
guess_coords_1 = resnet.predict(X_test) * 448.

resnet.load_weights('weights/best_resnet_loc_1.h5')
guess_coords_2 = resnet.predict(X_test) * 448.

resnet.load_weights('weights/best_resnet_loc_2.h5')
guess_coords_3 = resnet.predict(X_test) * 448.

resnet.load_weights('weights/best_resnet_loc_3.h5')
guess_coords_4 = resnet.predict(X_test) * 448.

resnet.load_weights('weights/best_resnet_loc_4.h5')
guess_coords_5 = resnet.predict(X_test) * 448.

guess_coords = (guess_coords_1 + guess_coords_2 + guess_coords_3 +
                guess_coords_4 + guess_coords_5) / 5.0

# some things to try
# 1) inflate the bounding box by some amount, say 10%, to allow for some error
# 2) force bounding box to be square, by taking the max of width and height and setting both to that
# 3)

choice_imgs = random.sample(xrange(len(X_test)), 10)

i = 0
for choice in choice_imgs:

    # display the test image and it's bounding box
    test_img = X_test[choice]
    test_coords = y_test[choice] * 448.
    test_img = test_img.transpose(1,2,0)

    guess = guess_coords[choice]
    # inflate bounding box by 10%
    guess[0] -= 0.5 * (0.1 * (guess[0] + guess[2]))
    guess[1] -= 0.5 * (0.1 * (guess[1] + guess[3]))
    guess[2] += 0.5 * (0.1 * (guess[0] + guess[2]))
    guess[3] += 0.5 * (0.1 * (guess[1] + guess[3]))

    fig, ax = plt.subplots(1)

    # display the image
    # back to RGB
    test_img = test_img[:, :, [2,1,0]]
    # add back in the imagenet means
    test_img[:, :, 0] += 103.939
    test_img[:, :, 1] += 116.779
    test_img[:, :, 2] += 123.68

    ax.imshow(test_img / 255.)

    # draw the true bounding box in yellow
    rect_tru = patches.Rectangle((test_coords[0], test_coords[1]), test_coords[2], test_coords[3],
                                  linewidth=2, edgecolor='y',facecolor='none')

    # draw the guess bounding box in red
    rect_guess = patches.Rectangle((guess[0], guess[1]), guess[2], guess[3],
                                    linewidth=2, edgecolor='r',facecolor='none')

    # add the rectangles
    ax.add_patch(rect_tru)
    ax.add_patch(rect_guess)

    # display
    #plt.show()
    plt.savefig('loc_test_imgs/fish_box_ensmb_' + str(i) + '.png')
    plt.clf()

    # to save the crops each img
    # from PIL import Image
    # im = Image.open('path/to/file.png')
    # im_cropped = im.crop((x,y,x+width,y+height)) x,y,width,height = coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]
    # im.save('path/to/file' + 'cropped.png')

    i += 1
