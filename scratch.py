
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import SGD, rmsprop
from keras.callbacks import ModelCheckpoint

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:150]:
    layer.trainable=False
  for layer in model.layers[150:]:
    layer.trainable=True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


base_model=InceptionV3(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(2048,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(2048,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(4,activation='softmax')(x) #final layer with softmax activation


model=Model(input=base_model.input,output=preds)

nb_train_samples = get_nb_files("Task8/train_dir")
nb_classes = len(glob.glob("Task8/train_dir" + "/*"))
nb_val_samples = get_nb_files("Task8/val_dir")
nb_epoch = int(100)


#train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

# data prep
train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=35,
      width_shift_range=0.3,
      height_shift_range=0.3,
      shear_range=0.3,
      zoom_range=0.3,
      horizontal_flip=True
  )

train_generator=train_datagen.flow_from_directory('Task8/train_dir/',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=128,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=35,
      width_shift_range=0.3,
      height_shift_range=0.3,
      shear_range=0.3,
      zoom_range=0.3,
      horizontal_flip=True
  )

validation_generator = test_datagen.flow_from_directory('Task8/val_dir',
												 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=128,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

filepath = "_{acc:.4f}_{loss:.4f}_{epoch:02d}_"
checkpoint = ModelCheckpoint("models/{}inv3.model".format(filepath, monitor=['val_acc', 'val_loss'], verbose=1, save_best_only=True, mode='max'))

history_tl = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
   	validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    callbacks=[checkpoint],
    class_weight='auto')
  
  # fine-tuning
setup_to_finetune(model)

history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    callbacks=[checkpoint],
    class_weight='auto')


