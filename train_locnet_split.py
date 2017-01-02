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

from keras.optimizers import SGD, Adam
from keras.layers import merge, Input, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.utils.layer_utils import print_summary

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

from utils import threaded_batch_iter_loc_class, bbox_tta

# set up training params
ITERS = 250
BATCHSIZE = 64
LR_SCHEDULE = {
     0: 0.0001,
    200: 0.00001,
}

# get number for ensemble
ensmb = int(sys.argv[1])

# number of TTA
num_crops = int(sys.argv[2])

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
    x = Dropout(p=0.5)(x)

    #x = Dense(1024, activation='relu', name='fc7')(x)
    #x = Dropout(p=0.5)(x)
    #x = Dense(1024, activation='relu', name='fc8')(x)
    #x = Dropout(p=0.5)(x)

    bbox_out = Dense(4, activation='sigmoid', name='coords')(x)
    class_out = Dense(8, activation='softmax', name='class')(x)

    model = Model([img_input], [bbox_out, class_out])

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
y_dat = np.load('data/cache/bbox_train_clean_coords_fullsize.npy')
y_labels = np.load('data/cache/bbox_train_clean_labels.npy')

# load file names
f = gzip.open('data/cache/file_names_train_clean.pklz', 'rb')
filez = pickle.load(f)
f.close()

print 'X shape:', X_dat.shape
print 'y shape:', y_dat.shape
print 'File length:', len(filez)


print 'Coords min max before'
print np.amax(y_dat)
print np.amin(y_dat)

y_dat[y_dat < 0.] = 0.
y_dat[y_dat > 1.] = 1.

print 'Coords min max after'
print np.amax(y_dat)
print np.amin(y_dat)

# Do preprocessing consistent with how it was done when the ImageNet images were
# used to originally train the model
# 'RGB'->'BGR'
X_dat = X_dat[:, :, :, [2,1,0]]
# subtract channels means from ImageNet
X_dat[:, :, :, 0] -= 103.939
X_dat[:, :, :, 1] -= 116.779
X_dat[:, :, :, 2] -= 123.68

y_labels = lblr.fit_transform(y_labels)

idx = np.random.permutation(len(X_dat))
X_dat = X_dat[idx]
y_dat = y_dat[idx]
y_labels = y_labels[idx]

#X_train, X_test, y_bb_train, y_bb_test, y_lbl_train, y_lbl_test = train_test_split(X_dat, y_dat, y_labels, test_size=0.2)

split = int(X_dat.shape[0] * 0.15)
X_train = X_dat[split:]
X_test = X_dat[:split]
y_bb_train = y_dat[split:]
y_bb_test = y_dat[:split]
y_lbl_train = y_labels[split:]
y_lbl_test = y_labels[:split]

print 'Train/val data shapes:', X_train.shape, X_test.shape
print 'Train/val bb shapes:', y_bb_train.shape, y_bb_test.shape
print 'Train/val label shapes:', y_lbl_train.shape, y_lbl_test.shape

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

print_summary(resnet.layers)

adam = Adam(lr=0.0001)
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
resnet.compile(loss=['mse', 'categorical_crossentropy'], optimizer=adam, loss_weights=[1.,0.25])

#resnet.fit(X_train, y_train, batch_size=64, nb_epoch=100, shuffle=True,
#           verbose=2, validation_data=(X_test, y_test), callbacks=callback_list)

# load batch_iter
batch_iter = threaded_batch_iter_loc_class(batchsize=BATCHSIZE)

print "Starting training ..."

train_loss = []
train_bb_loss = []
train_class_loss = []
valid_loss = []
valid_bb_loss = []
valid_class_loss = []
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
        batch_bb_loss = []
        batch_class_loss = []
        for X_batch, y_bb_batch, y_lbl_batch in batch_iter(X_train, y_bb_train, y_lbl_train):
            overall_loss, bbox_loss, class_loss = resnet.train_on_batch(X_batch, [y_bb_batch, y_lbl_batch])
            batch_loss.append(overall_loss)
            batch_bb_loss.append(bbox_loss)
            batch_class_loss.append(class_loss)

        train_loss.append(np.mean(batch_loss))
        train_bb_loss.append(np.mean(batch_bb_loss))
        train_class_loss.append(np.mean(batch_class_loss))
        v_o_loss, v_bb_loss, v_class_loss = resnet.evaluate(X_test, [y_bb_test, y_lbl_test], batch_size=BATCHSIZE, verbose = 0)
        valid_loss.append(v_o_loss)
        valid_bb_loss.append(v_bb_loss)
        valid_class_loss.append(v_class_loss)
        end = time.time() - start
        print epoch, '| tO:{:.3} | tBB:{:.3} | tC:{:.3} | vO:{:.3} | vBB:{:.3} | vC:{:.3} | T:{}'.format(train_loss[epoch],
                                                                                                         train_bb_loss[epoch],
                                                                                                         train_class_loss[epoch],
                                                                                                         v_o_loss,
                                                                                                         v_bb_loss,
                                                                                                         v_class_loss,
                                                                                                         int(end))

        if v_bb_loss < best_vl:
            best_vl = v_bb_loss
            resnet.save_weights('weights/best_resnet_loc_split_' + str(ensmb) + '.h5')
            best_epoch = epoch

        if v_bb_loss > best_vl:
            patience += 1
        else:
            patience = 0

        if patience >= 50:
            break

except KeyboardInterrupt:
    pass

print 'best epoch:', best_epoch

'''
------------------------------------------------------------------------------------------------
Test an image and see how it did
------------------------------------------------------------------------------------------------
'''
resnet.load_weights('weights/best_resnet_loc_split_' + str(ensmb) + '.h5')

#vgg_model.load_weights('weights/best_vgg_bb_model_loc_' + str(ensmb) + '.h5')
guess_coords = bbox_tta(resnet, X_test, num_tta=num_crops)


# some things to try
# 1) inflate the bounding box by some amount, say 10%, to allow for some error
# 2) force bounding box to be square, by taking the max of width and height and setting both to that
# 3)

choice_imgs = random.sample(xrange(len(X_test)), 10)

i = 0
for choice in choice_imgs:

    # display the test image and it's bounding box
    test_img = X_test[choice]
    #test_img = imread(test_file)

    test_coords = y_bb_test[choice]

    # put bounding box back into original size
    test_coords[0] *= test_img.shape[1]
    test_coords[1] *= test_img.shape[0]
    test_coords[2] *= test_img.shape[1]
    test_coords[3] *= test_img.shape[0]

    # get the guess coordinates
    guess = guess_coords[choice]

    # put bounding box back into original size
    guess[0] *= test_img.shape[1]
    guess[1] *= test_img.shape[0]
    guess[2] *= test_img.shape[1]
    guess[3] *= test_img.shape[0]

    # inflate bounding box by 10%
    guess[0] -= 0.5 * (0.1 * guess[2])
    guess[1] -= 0.5 * (0.1 * guess[3])
    guess[2] += 0.5 * (0.1 * guess[2])
    guess[3] += 0.5 * (0.1 * guess[3])

    fig, ax = plt.subplots(1)

    # display the image
    # add back in the imagenet means
    test_img[:, :, 0] += 103.939
    test_img[:, :, 1] += 116.779
    test_img[:, :, 2] += 123.68
    # back to RGB
    test_img = test_img[:, :, [2,1,0]]

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
    plt.savefig('loc_test_imgs/fish_box_resnet_split_' + str(i) + '.png')
    plt.clf()

    i += 1
