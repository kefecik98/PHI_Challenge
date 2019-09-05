from numba import cuda

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import preprocess_input
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet169
from keras import regularizers

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers[:FREEZE_LAYER]:
    layer.trainable = False
  for layer in base_model.layers[FREEZE_LAYER:]:
    layer.trainable = True
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  #model.compile(optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
#  x = BatchNormalization()(x)
  x = GlobalAveragePooling2D()(x)
  x = Dense(2*FC_SIZE, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x) #new FC layer, random init
  x = BatchNormalization()(x)
  x = Dense(FC_SIZE, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x) #new FC layer, random init
  x = BatchNormalization()(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
#  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
#     layer.trainable = False
#  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
#     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.nb_epoch)
  batch_size = int(args.batch_size)


  # setup model
  base_model = DenseNet169(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)
  filepath = "{acc:.4f}_{loss:.4f}_{epoch:02d}_"
  checkpoint = ModelCheckpoint("models/t6_{}dn121.model".format(filepath, monitor=['val_acc', 'val_loss'], verbose=1, save_best_only=True, mode='max'))


  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=20,
      width_shift_range=0.3,
      height_shift_range=0.3,
      shear_range=0.3,
      zoom_range=0.3,
      horizontal_flip=True
  )
  # test_datagen = ImageDataGenerator(
  #     preprocessing_function=preprocess_input,
  #     rotation_range=20,
  #     width_shift_range=0.3,
  #     height_shift_range=0.3,
  #     shear_range=0.3,
  #     zoom_range=0.3,
  #     horizontal_flip=True
  # )

  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # validation_generator = test_datagen.flow_from_directory(
  #   args.val_dir,
  #   target_size=(IM_WIDTH, IM_HEIGHT),
  #   batch_size=batch_size,
  # )



  # transfer learning
  setup_to_transfer_learn(model, base_model)

  steps_per_epoch = round(nb_train_samples/batch_size)
#  steps_per_epochv = round(nb_val_samples/batch_size)
  history_tl = model.fit_generator(
    train_generator,
    epochs=nb_epoch/2,
    steps_per_epoch=steps_per_epoch,#    validation_data=validation_generator, validation_steps=steps_per_epochv,
    callbacks=[checkpoint],
    class_weight='auto')

  # fine-tuning
  setup_to_finetune(model)

  history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nb_epoch*2, #     validation_data=validation_generator, validation_steps=steps_per_epochv,
    callbacks=[checkpoint],
    class_weight='auto')



if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir", default="Task6/train_dir1/")
  a.add_argument("--val_dir", default="Task6/val_dir/")
  a.add_argument("--nb_epoch", default=50)
  a.add_argument("--batch_size", default=64)
  a.add_argument("--output_model_file", default="Res.model")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()
  IM_WIDTH, IM_HEIGHT = 244, 244 #fixed size for InceptionV3 which was 299 by 299
  FC_SIZE = 2048
  FREEZE_LAYER = 120


  train(args)
