import sys
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

import keras.backend as K
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import merge, Input, Dropout
from keras.layers import Dropout, ZeroPadding2D
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.utils.layer_utils import print_summary
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.preprocessing import image

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

from utils import threaded_batch_iter_class_synth, bbox_tta

from skimage.io import imshow

# set up training params
ITERS = 200
BATCHSIZE = 64
LR_SCHEDULE = {
     0: 0.00001,
    10: 0.0001,
   150: 0.00001,
}

# get number for ensemble
ensmb = int(sys.argv[1])

# LabelBinarizer
lblr = LabelBinarizer()

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

    img_input = Input(shape=(224,224,3), name='Input', dtype='float32')

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

    #x = Dense(1024, activation='relu', name='fc7')(x)
    #x = Dropout(p=0.5)(x)
    #x = Dense(1024, activation='relu', name='fc8')(x)
    #x = Dropout(p=0.5)(x)

    class_out = Dense(8, activation='softmax', name='class')(x)

    model = Model(img_input, class_out)

    # load weights
    #f = h5py.File('data/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')
    model.load_weights('data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

    if K.backend() == 'theano':
        convert_all_kernels_in_model(model)

    return model


'''
------------------------------------------------------------------------------------------------
Load the images and split into  training and validation.
------------------------------------------------------------------------------------------------
'''

X_dat = np.load('data/cache/bbox_train_clean_imgs_fullsize.npy')
y_coords = np.load('data/cache/bbox_train_clean_coords_fullsize.npy')
y_labels = np.load('data/cache/bbox_train_clean_labels.npy')

# load file names
f = gzip.open('data/cache/file_names_train.pklz', 'rb')
filez = pickle.load(f)
f.close()

print 'Data shapes:', X_dat.shape, y_coords.shape, y_labels.shape


'''
------------------------------------------------------------------------------------------------
Compile and Train the Model
------------------------------------------------------------------------------------------------
'''
resnet = ResNet50()

adam = Adam(lr=0.0001)
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
resnet.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

lblr.fit(y_labels)
print y_labels

idx = np.random.permutation(len(X_dat))
X_dat = X_dat[idx]
y_coords = y_coords[idx]
y_labels = y_labels[idx]

#X_train, X_test, y_bb_train, y_bb_test, y_lbl_train, y_lbl_test = train_test_split(X_dat, y_dat, y_labels, test_size=0.2)

split = int(X_dat.shape[0] * 0.15)
X_train = X_dat[split:]
X_test = X_dat[:split]
y_bb_train = y_coords[split:]
y_bb_test = y_coords[:split]
y_lbl_train = y_labels[split:]
y_lbl_test = lblr.transform(y_labels[:split])

print 'Train/val data shapes:', X_train.shape, X_test.shape
print 'Train/val bb shapes:', y_bb_train.shape, y_bb_test.shape
print 'Train/val label shapes:', y_lbl_train.shape, y_lbl_test.shape

# Do preprocessing consistent with how it was done when the ImageNet images were
# used to originally train the model
# 'RGB'->'BGR'
X_test = X_test[:, :, :, [2,1,0]]
# subtract channels means from ImageNet
X_test[:, :, :, 0] -= 103.939
X_test[:, :, :, 1] -= 116.779
X_test[:, :, :, 2] -= 123.68

# load batch_iter
batch_iter = threaded_batch_iter_class_synth(batchsize=BATCHSIZE, lblr=lblr)

print "Starting training ..."
# batch iterator with 300 epoch

#vgg_model.fit(X_train, [y_bb_train, y_lbl_train], batch_size=BATCHSIZE, nb_epoch=ITERS,
#              validation_data=(X_test, [y_bb_test, y_lbl_test]), verbose=2)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
best_vl = 20.0
patience = 0
try:
    for epoch in range(ITERS):
        # change learning rate according to schedule
        if epoch in LR_SCHEDULE:
            K.set_value(adam.lr, LR_SCHEDULE[epoch])
        start = time.time()
        #loss = batch_iterator(x_train, y_train, 64, model)
        batch_loss = []
        batch_acc = []
        batch_class_loss = []
        for X_batch, y_batch in batch_iter(X_train, y_lbl_train, y_bb_train):
            loss, acc_t = resnet.train_on_batch(X_batch, y_batch)
            batch_loss.append(loss)
            batch_acc.append(acc_t)

        train_loss.append(np.mean(batch_loss))
        train_acc.append(np.mean(batch_acc))
        v_loss, v_acc = resnet.evaluate(X_test, y_lbl_test, batch_size=BATCHSIZE, verbose = 0)
        valid_loss.append(v_loss)

        end = time.time() - start
        print epoch, '| tL:{:.3} | tA:{:.3} | vL:{:.3} | vA:{:.3} | T:{}'.format(train_loss[epoch],
                                                                                 train_acc[epoch],
                                                                                 v_loss,
                                                                                 v_acc,
                                                                                 int(end))

        if v_loss < best_vl:
            best_vl = v_loss
            resnet.save_weights('weights/best_resnet_fullimg_class_' + str(ensmb) + '.h5')
            best_epoch = epoch

        if v_loss > best_vl:
            patience += 1
        else:
            patience = 0

        if patience >= 50:
            break

except KeyboardInterrupt:
    pass

print 'best epoch:', best_epoch
