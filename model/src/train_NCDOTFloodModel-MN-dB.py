#!/usr/bin/env python
# coding: utf-8

# To summarize what I'm doing, I'm looking at 3 approaches
# 1. supervised "uncoupled" classification from unsupervised feature extraction
#--> PCA analysis to extract 100 principal components, then kNN classification
#--> lowest amount of supervision; no supervised feature extraction, no explicit coupling of features to classes (kNN functions like a lookup table)
# 2. supervised "uncoupled" classification from features from weakly supervised feature extraction
#--> use convolution layers, global pooling, and dense layer with no activation to create embeddings from images
#--> weight the convolution layers using a loss function that just positions embeddings in embedding space such that embeddings
#--> (embeddings different from feature vectors becos dont change size with different sized images)
#--> medium supervision; weakly supervised feature extraction, no explicit coupling of features to classes (kNN functions like a lookup table)
#--> relative few parameters (~300,000), scales with convolution layer sizes
# 3. supervised "fully coupled" transfer-learned-features--to--class--mapping (this is Evan's model based on mobilenet)
#--> mobilenet feature extraction with distillation head (max pool), and classifying head (dense layer with dropout and kernel regularization),
#--> retrained with data but feature extractor layers keep imagenet weights
#--> highest amount of supervision; no supervised feature extraction, but explicit mapping of those features to classes by iteratively adjusting a model to do so
#--> most amount of parameters, but only classification parameters tuned

# this script is the latter approach
# Main changes made to Evan G's original:
# 1. changed from ipython to python script
# 2. implemented model checkpoint as h5, removed pb/tf-js outputs
# rationale for h5: https://tensorflow.rstudio.com/reference/keras/save_model_hdf5/
# 3. for train/test, switched from generators to arrays. wanted to keep equivalent workflow
# with the unsupervised model I'm making, which needs arrays cos its weird
# 4. I standardize imagery, not normalize
# 5. reorg as imports / functions / varables / execution (just a personal style thing)
# 6. Used .5 validation split rather than .2
# 7. exposed all the hyperparameters up-front
# 8. tested model in .predict() mode
# 9. tested the full ("load model from json / load h5 weights / load image from jpeg / make prediction") workflow
# 10. organized all the code as more modular-leaning function calls than api calls, cleaner in terms of zombie variables too
# 11. added option to bypass training
# 12. added confusion matrix calc and viz

##==============================================
# #IMPORTS
##==============================================


# import the general stuff
import os
# from os import getcwd
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
from glob import glob
import matplotlib.cm as cm
from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics import confusion_matrix
import seaborn as sns #extended functionality / style to matplotlib plots
from random import shuffle

#Set GPU to use
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import the tf stuff
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import model_from_json
from datetime import datetime

import keras_tuner as kt
import tensorflow.keras.backend as K

# import tensorflowjs as tfjs

#code for GPU mem growth
#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
#Check TensorFlow Version
# print('TF version: {}' .format(tf.__version__))
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

##==============================================
# #FUNCTIONS
##==============================================


# ##==============================================
# # #TRAINING
# ##==============================================

##==============================================
def model_builder(hp):

    imshape = (height_width,height_width,3)

    # #base_model, no top layer, w/ imagenet weights
    # base_model = tf.keras.applications.MobileNetV2(input_shape = imshape,
    #                                              include_top = False,
    #                                              weights = None)

    #base_model, no top layer, w/ imagenet weights
    base_model = tf.keras.applications.MobileNetV2(input_shape = imshape,
                                                 include_top = False,
                                                 weights = 'imagenet')

    if SCRATCH:
        base_model.trainable = True
    else:
        base_model.trainable = False
    # base_model.summary()

    # add a new classifcation head

    final_layer = base_model.get_layer('out_relu')
    #print('shape of last layer is ', final_layer.output_shape)
    final_base_output = final_layer.output

    #add the last layer
    # GL.Av Pool
    #x = layers.Flatten()(final_base_output )
    x = layers.GlobalAveragePooling2D()(final_base_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation

    hp_units = hp.Int('units', min_value=256, max_value=768, step=64)

    x = layers.Dense(hp_units, activation='relu')(x) #512
    # Add a dropout rate of 0.5

    dropout_rate = hp.Choice('dropout_rate', values=[.2,.25,.3,.35,.4,.45,.5])

    x = layers.Dropout(dropout_rate)(x)
    # Add a final sigmoid layer for classification
    if mode=='binary':
        x = layers.Dense(1, activation='sigmoid')(x)
    else:
        x = layers.Dense(NCLASSES, activation='sigmoid')(x)

    model = Model(base_model.input, x)

    hp_learning_rate = hp.Choice('learning_rate', values=[.5e-4, 1e-5, .5e-5,1e-6, .5e-6])

    if mode is 'binary':
        acc_metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')

        #build the model
        model.compile(loss = 'binary_crossentropy',
                      optimizer = tf.keras.optimizers.Adam(lr = hp_learning_rate), #lr),
                      metrics = acc_metric)
    else:

        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            name='categorical_crossentropy')

        model.compile(optimizer=tf.keras.optimizers.Adam(lr= hp_learning_rate), #lr),
                      loss=loss_fn,
                      metrics=['accuracy'])

    return model



##==============================================
def model_trainer(model,aug=False):

    # set checkpoint file
    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_loss',
                                    verbose=0, save_best_only=True, mode='min',
                                    save_weights_only = True)

    callbacks = [earlystop,model_checkpoint]

    if mode is 'binary':

        if aug:
            history = model.fit(img_generator, epochs=EPOCHS, batch_size=BS, callbacks=callbacks, validation_data=val_generator)

        else:
            history = model.fit(
                x_train,
                y_train,
                batch_size=BS,
                epochs=EPOCHS,
                # We pass some validation for
                # monitoring validation loss and metrics
                # at the end of each epoch
                validation_data=(x_test, y_test),
                callbacks =[callbacks])
    else:

        if aug:
            history = model.fit(img_generator, epochs=EPOCHS, batch_size=BS, callbacks=callbacks, validation_data=val_generator)

        else:
            history = model.fit(
                x_train,
                tf.one_hot(y_train,NCLASSES),
                batch_size=BS,
                epochs=EPOCHS,
                # We pass some validation for
                # monitoring validation loss and metrics
                # at the end of each epoch
                validation_data=(x_test, tf.one_hot(y_test,NCLASSES)),
                callbacks =[callbacks])

    #look at the metrics from training
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    # plt.show()
    plt.savefig(weights_file.replace('.h5','_acc.png'), dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, 'r--', label='Training loss')
    plt.plot(epochs, val_loss, 'b--', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    # plt.show()
    plt.savefig(weights_file.replace('.h5','_loss.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    return model


##==============================================
def get_data(train_files,test_files,height_width):
    '''
    read and resize and standardize imager from files to numpy arrays
    '''
    x_train = np.zeros((len(train_files),height_width,height_width,3))
    for counter,f in enumerate(train_files):
        im = resize(imread(f), (height_width,height_width))
        x_train[counter]=standardize(im)
    x_train = x_train.astype("float32")

    x_test = np.zeros((len(test_files),height_width,height_width,3))
    for counter,f in enumerate(test_files):
        im = resize(imread(f), (height_width,height_width))
        x_test[counter]=standardize(im)

    x_test = x_test.astype("float32")

    y_train = []
    for f in train_files:
        y_train.append(class_dict[f.split(os.sep)[-1].split('_X_')[0]])

    y_train = np.expand_dims(y_train,-1).astype('uint8')
    y_train = np.squeeze(y_train)

    y_test = []
    for f in test_files:
        y_test.append(class_dict[f.split(os.sep)[-1].split('_X_')[0]])
    y_test = np.expand_dims(y_test,-1).astype('uint8')

    y_test = np.squeeze(y_test)
    return x_train, y_train, x_test, y_test

##==============================================
# #PREDICTION
##==============================================
def predict_flood(f,FloodCAMML,height_width=224):
    '''
    given file path string, f, and height_width (default=224)
    return 0 for no flood and 1 for flood
    '''
    img = resize(imread(f), (height_width,height_width))
    Orimg = imread(f)
    img_array = np.expand_dims(img,axis=0)

    try:
        do_gradcam_viz(img, FloodCAMML, outfile=f.replace('.jpg','_gradcam.png'))
    except:
        pass
    return FloodCAMML.predict(img_array, batch_size = 1, verbose = False)

##==============================================
def get_model_load_weights(weights_file):
    '''
    given file path string, weights file, loads keras model and assigns weights
    '''
    json_file = open(weights_file.replace('.h5','.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FloodCAMML = model_from_json(loaded_model_json)
    #print("Loading weights into model")
    # load weights into new model
    FloodCAMML.load_weights(weights_file)
    return FloodCAMML

# #IMG UTILS
##==============================================
def standardize(img):
    '''
    standardization using adjusted standard deviation,
    then rescales an input dat between mn and mx
    '''
    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    img = rescale(img, 0, 1)
    del m, s, N
    if np.ndim(img)!=3:
        img = np.dstack((img,img,img))

    return img

##==============================================
def rescale(dat,mn,mx):
    '''
    rescales an input dat between mn and mx
    '''
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn


#from https://keras.io/examples/vision/grad_cam/
##==============================================
# #GRADCAM
##==============================================
def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer

    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    #grads = tape.gradient(bottom_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

##==============================================
def do_gradcam_viz(img, model, outfile):
    #define last conv layers
    last_conv_layer_name = "out_relu"
    classifier_layer_names = [
        "global_average_pooling2d",
        "dense",
        "dropout",
        "dense_1",
    ]

    alpha=0.3

    img_array = np.expand_dims(img,axis=0)

    #from https://keras.io/examples/vision/grad_cam/
    # Make the heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
    except:

        classifier_layer_names = [
            "global_average_pooling2d_1",
            "dense_2",
            "dropout_1",
            "dense_3",
        ]
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)

    # rescale image (range 0-255)
    heatmap = np.uint8(255 * heatmap)

    # use viridis for heatmap
    vir = cm.get_cmap("viridis")
    vir_colors = vir(np.arange(256))[:, :3]
    vir_heatmap = vir_colors[heatmap]

    # make the heatmap
    vir_heatmap = tf.keras.preprocessing.image.array_to_img(vir_heatmap)
    vir_heatmapI = vir_heatmap.resize((Orimg.shape[1], Orimg.shape[0]))
    vir_heatmap = tf.keras.preprocessing.image.img_to_array(vir_heatmapI)

    #put heatmpa on image
    superimposed_img = vir_heatmap * alpha + Orimg
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Image, heatmap and overlay
    # Display heatmap
    plt.figure(figsize=(20,10))
    plt.subplot(121)#31)
    plt.imshow(Orimg) ; plt.title('Orig image')
    #tf.keras.preprocessing.image.load_img(impath, target_size = imsize))
    # plt.subplot(132)
    # plt.imshow(vir_heatmapI); plt.title('heatmap')
    plt.subplot(122)#33)
    plt.imshow(superimposed_img); plt.title('Superimposed GRADCAM')

    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


##==============================================
# #INPUTS
##=============================================
# root = '/home/marda/Downloads/HX_Ted_2020_NCTC/All_Photos/Recoded'
# root = '/home/marda/Downloads/FloodCamML/Data/v1_20210709'
root = '/home/marda/Downloads/FloodCamML/Data/v2_20210712'

mode='categorical'
# mode = 'binary'

if mode == 'binary':
    #put names in alphabetical order
    CLASSES = ['No_water','Water']
    class_dict={'No_water':0, 'Water':1}
    all_files = glob(root+'/Water/*.jpg')+glob(root+'/No_water/*.jpg')
    total_files = len(all_files)
else:
    #put names in alphabetical order
    # CLASSES = ['Not_usable','No_water','Not_sure','Water']
    # class_dict={'Not_usable':0, 'No_water':1, 'Not_sure':2,  'Water':3}
    # all_files = glob(root+'/Water/*.jpg')+glob(root+'/No_water/*.jpg')+glob(root+'/Not_sure/*.jpg')+glob(root+'/Not_usable/*.jpg')

    CLASSES = ['Not_usable','No_water','Water']
    class_dict={'Not_usable':0, 'No_water':1, 'Water':2}
    all_files = glob(root+'/Water/*.jpg')+glob(root+'/No_water/*.jpg')+glob(root+'/Not_usable/*.jpg')

    total_files = len(all_files)

NCLASSES = len(CLASSES)



# DOTRAIN = False
DOTRAIN = True
SCRATCH = True # False
# alpha=0.4
height_width = 224
BS = 12# 20 #16 #12
EPOCHS = 200
PATIENCE = 25
VAL_SPLIT = 0.7 #5


shuffle(all_files)
nn = int(VAL_SPLIT*total_files)

test_files = all_files[:nn] #test first

# train is whatever is left
train_files = all_files[nn:]


print('%i train files' % (len(train_files)))
print('%i test files' % (len(test_files)))


# DROPOUT = 0.4#5
# NUM_NEURONS = 128 #512
# if SCRATCH:
#     lr = 1e-5
# else:
#     lr = 1e-4
#========================================================


##==============================================
# #PREP DATA
##==============================================
# if DOTRAIN:
x_train, y_train, x_test, y_test = get_data(train_files,test_files,height_width)

# x = x_train[:BS]
# y = y_train[:BS]

plt.figure(figsize=(15, 15))
counter = 1
for im, l in zip(x_train[:BS], y_train[:BS]):
    if BS==16:
        ax = plt.subplot(4, 4,counter)
    else:
        ax = plt.subplot(4, 5,counter)

    plt.imshow(im)

    # if mode=='binary':
    l=np.int(l)
    plt.title(CLASSES[l], fontsize=9)
    # else:
    #     plt.title(CLASSES[np.argmax(l)], fontsize=9)

    plt.axis('off')
    counter +=1

# plt.show()
plt.savefig('NCDot_example_train_batch_'+mode+'_'+str(NCLASSES)+'.png',dpi=300,bbox_inches='tight')
plt.close()
# del x, y

#hp=kt.HyperParameters()
# Variation of HyperBand algorithm.
#
# Reference:
#     Li, Lisha, and Kevin Jamieson.
#     ["Hyperband: A Novel Bandit-Based
#      Approach to Hyperparameter Optimization."
#     Journal of Machine Learning Research 18 (2018): 1-52](
#         http://jmlr.org/papers/v18/16-558.html).

tuner = kt.Hyperband(model_builder,
                     objective='val_loss', #'val_accuracy',
                     max_epochs=int(EPOCHS/2),
                     factor=3,
                     hyperband_iterations=3,
                     seed=2021,
                     directory='tuner',
                     project_name='CamML4class_'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=PATIENCE)

tuner.search(x_train, tf.one_hot(y_train,NCLASSES), epochs=EPOCHS, validation_data=(x_test, tf.one_hot(y_test,NCLASSES)), callbacks=[earlystop])

# models = tuner.get_best_models(num_models=2)

tuner.results_summary()

K.clear_session()

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
optimal number of units in the first densely-connected layer = {best_hps.get('units')} \n
optimal learning rate for the optimizer is {best_hps.get('learning_rate')}\n
optimal dropout rate for the optimizer is {best_hps.get('dropout_rate')}\n
""")

#
# ##==============================================
# # #BUILD MODEL
# ##==============================================

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

DROPOUT = best_hps.get('dropout_rate')
NUM_NEURONS = best_hps.get('units')
lr = best_hps.get('learning_rate')

try:
    os.mkdir('results')
except:
    pass

if SCRATCH:
    #tl=transferlarning, mv2=MobileNetV2 feature extractor
    weights_file = 'results/scratch_mv2_bs'+str(BS)+'_drop'+str(DROPOUT)+'_nn'+str(NUM_NEURONS)+'_sz'+str(height_width)+'_lr'+str(lr)+'_val'+str(VAL_SPLIT)+'.h5'

else:
    #tl=transferlarning, mv2=MobileNetV2 feature extractor
    weights_file = 'results/tl_mv2_bs'+str(BS)+'_drop'+str(DROPOUT)+'_nn'+str(NUM_NEURONS)+'_sz'+str(height_width)+'_lr'+str(lr)+'_val'+str(VAL_SPLIT)+'.h5'

print(weights_file)

# ##==============================================
# # #TRAIN MODEL
# ##==============================================
# if DOTRAIN:
#     model = train_model(model,PATIENCE,weights_file,BS,EPOCHS,img_generator,val_generator,mode)
model = model_trainer(model)

plt.close('all')

model_json = model.to_json()
with open(weights_file.replace('.h5','.json'), "w") as json_file:
    json_file.write(model_json)

# save as TF.js
# tfjs.converters.save_keras_model(model, './JSmodel')

model.save('Rmodel_'+weights_file.replace('.h5','')+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

K.clear_session()

##==============================================
# #GRADCAM VIZ
##==============================================
notsure_files = all_files[:BS]
# if DOTRAIN:

try:
    os.mkdir('predict')
except:
    pass

for k in np.arange(BS):
    use = np.random.randint(len(notsure_files))### 42
    f = notsure_files[use]
    img = resize(imread(f), (height_width,height_width))
    Orimg = imread(f)
    outfile = 'predict'+os.sep+f.split(os.sep)[-1].replace('.jpg','_gradcam.png')

    try:
        do_gradcam_viz(img, model, outfile)
    except:
        pass
plt.close('all')

##==============================================
# #CONFUSION MATRUX
##==============================================
# if DOTRAIN:
y_pred = model.predict(x_test).squeeze()

if mode=='binary':
    y_pred = (y_pred>.5).astype(int)
else:
    y_pred = np.argmax(y_pred, -1)

cmat = confusion_matrix(y_test, y_pred)
cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,8))
sns.heatmap(cmat,
  annot=True,
  cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))

tick_marks = np.arange(len(CLASSES))+.5
plt.xticks(tick_marks, [c for c in CLASSES], rotation=90,fontsize=12)
plt.yticks(tick_marks, [c for c in CLASSES],rotation=0, fontsize=12)
plt.title('N = '+str(len(y_test)), fontsize=12)

plt.savefig(weights_file.replace('.h5','_cm.png'), dpi=200, bbox_inches='tight')
plt.close()


##==============================================
# #PREDICTION SIMULATION
##==============================================
plt.figure(figsize=(16,16))
counter=1
for k in x_test[:16]:
    plt.subplot(4,4,counter)
    plt.imshow(k)
    plt.axis('off')
    y = model.predict(np.expand_dims(k,0))[0]
    plt.title('%s (prob = %f)' % ( CLASSES[np.argmax(y)] , y.max() ), fontsize=7)
    counter+=1
plt.savefig(weights_file.replace('.h5','_sampletest.png'), dpi=200, bbox_inches='tight')
plt.close()

plt.close('all')

#
#
# #####=================================================================
# #====================================================================
# ## AUG<EMTIAION
#
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         featurewise_center=True,
#         featurewise_std_normalization=True,
#         zoom_range=0.1,
#         shear_range=0,
#         vertical_flip = False,
#         rotation_range=5,
#         horizontal_flip=True,
#         fill_mode='reflect',
#         validation_split = VAL_SPLIT)
#
#
# img_generator = datagen.flow_from_directory(
#         root,
#         subset='training',
#         target_size=(height_width, height_width),
#         batch_size=BS,
#         shuffle=True,
#         class_mode=['binary' if mode=='binary' else 'categorical'][0],
#         seed=2021)
#
#
# val_generator = datagen.flow_from_directory(
#         root,
#         subset='validation',
#         target_size=(height_width, height_width),
#         batch_size=BS,
#         shuffle=True,
#         class_mode=['binary' if mode=='binary' else 'categorical'][0],
#         seed=2021)
#
# x, y = next(img_generator)
#
# plt.figure(figsize=(15, 15))
# counter = 1
# for im, l in zip(x,y):
#     ax = plt.subplot(4, 4,counter)
#     plt.imshow(im/255.)
#
#     if mode=='binary':
#         l=np.int(l)
#         plt.title(CLASSES[l], fontsize=9)
#     else:
#         plt.title(CLASSES[np.argmax(l)], fontsize=9)
#
#     plt.axis('off')
#     counter +=1
#
# # plt.show()
# plt.savefig('NCDot_example_train_augbatch_'+mode+'.png',dpi=300,bbox_inches='tight')
# plt.close()
#
#
# # Build the model with the optimal hyperparameters
# model_aug = tuner.hypermodel.build(best_hps)
#
# DROPOUT = best_hps.get('dropout_rate')
# NUM_NEURONS = best_hps.get('units')
# lr = best_hps.get('learning_rate')
#
# if SCRATCH:
#     #tl=transferlarning, mv2=MobileNetV2 feature extractor
#     weights_file = 'aug_scratch_mv2_bs'+str(BS)+'_drop'+str(DROPOUT)+'_nn'+str(NUM_NEURONS)+'_sz'+str(height_width)+'_lr'+str(lr)+'_val'+str(VAL_SPLIT)+'.h5'
#
# else:
#     #tl=transferlarning, mv2=MobileNetV2 feature extractor
#     weights_file = 'aug_tl_mv2_bs'+str(BS)+'_drop'+str(DROPOUT)+'_nn'+str(NUM_NEURONS)+'_sz'+str(height_width)+'_lr'+str(lr)+'_val'+str(VAL_SPLIT)+'.h5'
#
# print(weights_file)
#
# # ##==============================================
# # # #TRAIN MODEL
# # ##==============================================
# model = model_trainer(model,aug=True)
#
# model_json = model.to_json()
# with open(weights_file.replace('.h5','.json'), "w") as json_file:
#     json_file.write(model_json)
#
# # save as TF.js
# # tfjs.converters.save_keras_model(model, './JSmodel')
#
# model.save('Rmodel_'+weights_file.replace('.h5','')+datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
#
#
# y_pred = model.predict(x_test).squeeze()
#
# if mode=='binary':
#     y_pred = (y_pred>.5).astype(int)
# else:
#     y_pred = np.argmax(y_pred, -1)
#
# cmat = confusion_matrix(y_test, y_pred)
# cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
#
# plt.figure(figsize=(8,8))
# sns.heatmap(cmat,
#   annot=True,
#   cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))
#
# tick_marks = np.arange(len(CLASSES))+.5
# plt.xticks(tick_marks, [c for c in CLASSES], rotation=90,fontsize=12)
# plt.yticks(tick_marks, [c for c in CLASSES],rotation=0, fontsize=12)
# plt.title('N = '+str(len(y_test)), fontsize=12)
#
# plt.savefig(weights_file.replace('.h5','_cm.png'), dpi=200, bbox_inches='tight')
# plt.close()
#


##################################################

#
# tuner = kt.Hyperband(model_builder,
#                      objective='val_accuracy',
#                      max_epochs=10,
#                      factor=3,
#                      directory='tuner',
#                      project_name='CamML-aug')
#
# earlystop = EarlyStopping(monitor="val_loss",
#                               mode="min", patience=PATIENCE)
#
# tuner.search(img_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=[earlystop])
#
# # models = tuner.get_best_models(num_models=2)
#
# tuner.results_summary()
#
#
# # Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#
# print(f"""
# optimal number of units in the first densely-connected layer = {best_hps.get('units')} \n
# optimal learning rate for the optimizer is {best_hps.get('learning_rate')}\n
# optimal dropout rate for the optimizer is {best_hps.get('dropout_rate')}\n
# """)




#
# # Build the model with the optimal hyperparameters
# model_aug = tuner.hypermodel.build(best_hps)
#
# DROPOUT = best_hps.get('dropout_rate')
# NUM_NEURONS = best_hps.get('units')
# lr = best_hps.get('learning_rate')
#
# if SCRATCH:
#     #tl=transferlarning, mv2=MobileNetV2 feature extractor
#     weights_file = 'aug-tuner_scratch_mv2_bs'+str(BS)+'_drop'+str(DROPOUT)+'_nn'+str(NUM_NEURONS)+'_sz'+str(height_width)+'_lr'+str(lr)+'_val'+str(VAL_SPLIT)+'.h5'
#
# else:
#     #tl=transferlarning, mv2=MobileNetV2 feature extractor
#     weights_file = 'aug-tuner_tl_mv2_bs'+str(BS)+'_drop'+str(DROPOUT)+'_nn'+str(NUM_NEURONS)+'_sz'+str(height_width)+'_lr'+str(lr)+'_val'+str(VAL_SPLIT)+'.h5'
#
# print(weights_file)

# ##==============================================
# # #TRAIN MODEL
# ##==============================================
# model = model_trainer(model,aug=True)
#
# model_json = model.to_json()
# with open(weights_file.replace('.h5','.json'), "w") as json_file:
#     json_file.write(model_json)
#
# # save as TF.js
# # tfjs.converters.save_keras_model(model, './JSmodel')
#
# model.save('Rmodel_'+weights_file.replace('.h5','')+datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
#
#
# y_pred = model.predict(x_test).squeeze()
#
# if mode=='binary':
#     y_pred = (y_pred>.5).astype(int)
# else:
#     y_pred = np.argmax(y_pred, -1)
#
# cmat = confusion_matrix(y_test, y_pred)
# cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
#
# plt.figure(figsize=(8,8))
# sns.heatmap(cmat,
#   annot=True,
#   cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))
#
# tick_marks = np.arange(len(CLASSES))+.5
# plt.xticks(tick_marks, [c for c in CLASSES], rotation=90,fontsize=12)
# plt.yticks(tick_marks, [c for c in CLASSES],rotation=0, fontsize=12)
# plt.title('N = '+str(len(y_test)), fontsize=12)
#
# plt.savefig(weights_file.replace('.h5','_cm.png'), dpi=200, bbox_inches='tight')
# plt.close()
#



# def build_tl_mv2_model(hp,height_width,NUM_NEURONS,DROPOUT,lr,SCRATCH, mode):
#     '''
#     mobilenet feature extraction with distillation head (max pool), and classifying head (dense layer with dropout and kernel regularization),
#     '''
#     imshape = (height_width,height_width,3)
#
#     # define the metrics
#     #load or build model
#     ##########
#
#     #base_model, no top layer, w/ imagenet weights
#     base_model = tf.keras.applications.MobileNetV2(input_shape = imshape,
#                                                  include_top = False,
#                                                  weights = 'imagenet')
#
#     if SCRATCH:
#         base_model.trainable = True
#     else:
#         base_model.trainable = False
#     # base_model.summary()
#
#     # add a new classifcation head
#
#     final_layer = base_model.get_layer('out_relu')
#     print('shape of last layer is ', final_layer.output_shape)
#     final_base_output = final_layer.output
#
#     #add the last layer
#     # GL.Av Pool
#     #x = layers.Flatten()(final_base_output )
#     x = layers.GlobalAveragePooling2D()(final_base_output)
#     # Add a fully connected layer with 1,024 hidden units and ReLU activation
#
#     x = layers.Dense(128, activation='relu')(x) #512
#     # Add a dropout rate of 0.5
#     x = layers.Dropout(DROPOUT)(x)
#     # Add a final sigmoid layer for classification
#     if mode=='binary':
#         x = layers.Dense(1, activation='sigmoid')(x)
#     else:
#         x = layers.Dense(3, activation='sigmoid')(x)
#
#     model = Model(base_model.input, x)
#
#     if mode is 'binary':
#         acc_metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')
#
#         #build the model
#         model.compile(loss = 'binary_crossentropy',
#                       optimizer = tf.keras.optimizers.Adam(lr = lr), #lr),
#                       metrics = acc_metric)
#     else:
#
#         loss_fn = tf.keras.losses.CategoricalCrossentropy(
#             from_logits=False,
#             name='categorical_crossentropy')
#
#         model.compile(optimizer=tf.keras.optimizers.Adam(lr= lr), #lr),
#                       loss=loss_fn,
#                       metrics=['accuracy'])
#
#
#     return model

#
# ##==============================================
# def train_model(model,PATIENCE,weights_file,BS,EPOCHS,img_generator,val_generator,mode, NCLASSES):
#     '''
#     train model with early stopping, h5 checkpoint, and plot acc and loss curves
#     '''
#     earlystop = EarlyStopping(monitor="val_loss",
#                                   mode="min", patience=PATIENCE)
#
#     # set checkpoint file
#     model_checkpoint = ModelCheckpoint(weights_file, monitor='val_loss',
#                                     verbose=0, save_best_only=True, mode='min',
#                                     save_weights_only = True)
#
#     callbacks = [earlystop,model_checkpoint]
#
#     if mode is 'binary':
#         # history = model.fit(img_generator, epochs=EPOCHS, batch_size=BS, callbacks=callbacks, validation_data=val_generator)
#
#         history = model.fit(
#             x_train,
#             y_train,
#             batch_size=BS,
#             epochs=EPOCHS,
#             # We pass some validation for
#             # monitoring validation loss and metrics
#             # at the end of each epoch
#             validation_data=(x_test, y_test),
#             callbacks =[callbacks]
#         )
#     else:
#
#         history = model.fit(
#             x_train,
#             tf.one_hot(y_train,NCLASSES),
#             batch_size=BS,
#             epochs=EPOCHS,
#             # We pass some validation for
#             # monitoring validation loss and metrics
#             # at the end of each epoch
#             validation_data=(x_test, tf.one_hot(y_test,NCLASSES)),
#             callbacks =[callbacks]
#         )
#         # history = model.fit(img_generator, epochs=EPOCHS, batch_size=BS, callbacks=callbacks, validation_data=val_generator)
#
#     #look at the metrics from training
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     epochs = range(len(acc))
#
#     plt.figure()
#     plt.plot(epochs, acc, 'r', label='Training accuracy')
#     plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#     plt.title('Training and validation accuracy')
#     plt.legend(loc=0)
#     # plt.show()
#     plt.savefig(weights_file.replace('.h5','_acc.png'), dpi=200, bbox_inches='tight')
#     plt.close()
#
#     plt.figure()
#     plt.plot(epochs, loss, 'r--', label='Training loss')
#     plt.plot(epochs, val_loss, 'b--', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend(loc=0)
#     # plt.show()
#     plt.savefig(weights_file.replace('.h5','_loss.png'), dpi=200, bbox_inches='tight')
#     plt.close()
#
#     return model


# if DOTRAIN:
#     model = build_tl_mv2_model(height_width,NUM_NEURONS,DROPOUT,lr, SCRATCH, mode)
#     #model.summary()
#
# FloodCAMML = get_model_load_weights(weights_file)
# use = np.random.randint(100)### 42
# # f = notsure_files[use]
# f = sample_files[use]
#
# print("=================================")
# print("=================================")

# flood = predict_flood(f,FloodCAMML,height_width)
# if flood>0.5:
#     print("flood")
# else:
#     print("no flood")
# print("=================================")


#
#
# total_train = int(total_files*VAL_SPLIT)
# total_val = int(total_files*VAL_SPLIT)

# notsure_files = glob(root+'/Not_sure*.jpg')
#
# sample_files = notsure_files
#
# print('%i not sure files' % (len(notsure_files)))


# sample_files = glob('../../../HX_Ted_2020_NCTC/Buxton/*.jpg')+glob('../../../HX_Ted_2020_NCTC/Canal/*.jpg')+glob('../../../HX_Ted_2020_NCTC/Mirlo/*.jpg')+glob('../../../HX_Ted_2020_NCTC/Ocracoke/*.jpg')
# 2341 files
#
# CLASSES = ['Buxton', 'Canal', 'Mirlo', 'Ocracoke']
# class_dict={'Buxton':0,  'Canal':1,  'Mirlo':2, 'Ocracoke':3 }
# site_code_train = np.ones(len(train_files))*99
# for counter,t in enumerate(train_files):
#     for k in class_dict.keys():
#         if k in t:
#             site_code_train[counter]=class_dict[k]



##==============================================
# #CODE IN EVAN G'S (WICKED) IMPLEMENTATION NOT USED
##==============================================

#===========================================================
# def get_img_array(img_path, size):
#     # `img` is a PIL image of size 299x299
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size = imsize)
#     # `array` is a float32 Numpy array of shape (299, 299, 3)
#     array = tf.keras.preprocessing.image.img_to_array(img)
#     # We add a dimension to transform our array into a "batch"
#     # of size (1, 299, 299, 3)
#     array = np.expand_dims(array, axis=0)
#     return array

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
#
#
# # In[3]:
#
#
# #Check for GPU utilization
# if tf.test.gpu_device_name():
#     print(' GPU: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("No GPU")

#
# #build data generators for training and validaton
#
# split = 0.2
#
# total_train = 320
# total_val = 78
#
#
# # Define dirs and files
# train_data_dir = '../data/SunnyD'
#
# # Add our data-augmentation parameters to ImageDataGenerator and split data
# train_datagen = ImageDataGenerator(rescale =1./255.,
#                                    rotation_range = 45,
#                                    width_shift_range = 0.2,
#                                    height_shift_range = 0.2,
#                                    shear_range = 0.4,
#                                    horizontal_flip = True,
#                                    vertical_flip = True,
#                                    validation_split = split)
#
#
# #set batch Size
# batch_size = 16
#
# #set Image size (RGB so imshape is 3)
# pix_dim = 224
# imsize = (pix_dim,pix_dim)
# imshape = (pix_dim,pix_dim,3)
#
# # Flow training images in batches
# train_generator = train_datagen.flow_from_directory(train_data_dir,
#                                                     batch_size = batch_size,
#                                                     class_mode = 'binary',
#                                                     target_size = imsize,
#                                                     subset='training')
#
# # Flow validation images in batches
# validation_generator =  train_datagen.flow_from_directory(train_data_dir, # same directory as training data,
#                                                         batch_size = batch_size,
#                                                         class_mode = 'binary',
#                                                         target_size = imsize,
#                                                         subset='validation')


# In[6]:


# In[19]:

#
# #save the model
# filepath = './models/MN2_model_TB'
# model.save(filepath)
#
# #load model
#model = tf.keras.models.load_model(filepath, compile = True)


#changing dropot rate or dense nodes doesnt help. try different loss


#
# #train the model
# history = model.fit(train_generator,
#                     steps_per_epoch = total_train // batch_size,
#                     validation_data = validation_generator,
#                     epochs= 20,
#                     validation_steps =  total_val // batch_size,
#                     callbacks =[callbacks])


# In[11]:



# In[18]:


# save as TF.js
# import tensorflowjs as tfjs
#
# tfjs.converters.save_keras_model(model, './models')


# In[ ]:

# # get the original ERI image
# Orimg = tf.keras.preprocessing.image.load_img(impath, target_size = imsize)
# Orimg = tf.keras.preprocessing.image.img_to_array(Orimg)

# Prepare image
# impath = "../data/SunnyD/flood/2020-09-20_07_15_23.292572-Mirlo.jpg"
# impath = "../data/SunnyD/flood/2020-09-20_07_35_33.722493-Ocracoke.jpg"
# impath = "../data/SunnyD/flood/2020-09-20_09_25_50.967305-Canal.jpg"
# impath = "../data/SunnyD/flood/2020-09-20_13_26_31.136067-Buxton.jpg"
# img = tf.keras.preprocessing.image.load_img(f,target_size = height_width)
# img = tf.keras.preprocessing.image.img_to_array(img)
# img = img/255
# img_array = np.expand_dims(img,axis=0)
