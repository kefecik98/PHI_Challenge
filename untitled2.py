
from numba import cuda

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import preprocess_input


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
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args, arch):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  #nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.nb_epoch)
  #batch_size = int(args.batch_size)
  
  if arch == 'vgg19':
      from keras.applications.vgg19 import VGG19
      # setup model
      base_model = VGG19(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
      model = add_new_last_layer(base_model, nb_classes)
      
      filepath = "_{acc:.4f}_{loss:.4f}_{epoch:02d}_"
      checkpoint = ModelCheckpoint("D:/CODE/New folder/{}vgg19.model".format(filepath, monitor=['acc', 'loss'], verbose=1, save_best_only=True, mode='max'))

  elif arch == 'vgg16':
      from keras.applications.vgg16 import VGG16
      # setup model
      base_model = VGG16(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
      model = add_new_last_layer(base_model, nb_classes)
      
      filepath = "_{acc:.4f}_{loss:.4f}_{epoch:02d}_"
      checkpoint = ModelCheckpoint("D:/CODE/New folder/{}vgg16.model".format(filepath, monitor=['acc', 'loss'], verbose=1, save_best_only=True, mode='max'))

  elif arch == 'xception':
      from keras.applications.xception import Xception
      # setup model
      base_model = Xception(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
      model = add_new_last_layer(base_model, nb_classes)
      
      filepath = "_{acc:.4f}_{loss:.4f}_{epoch:02d}_"
      checkpoint = ModelCheckpoint("D:/CODE/New folder/{}xception.model".format(filepath, monitor=['acc', 'loss'], verbose=1, save_best_only=True, mode='max'))

  elif arch == 'res50':
      from keras.applications.resnet50 import ResNet50
      # setup model
      base_model = ResNet50(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
      model = add_new_last_layer(base_model, nb_classes)
      
      filepath = "_{acc:.4f}_{loss:.4f}_{epoch:02d}_"
      checkpoint = ModelCheckpoint("D:/CODE/New folder/{}res50.model".format(filepath, monitor=['acc', 'loss'], verbose=1, save_best_only=True, mode='max'))

  elif arch == 'inv3':
      from keras.applications.inception_v3 import InceptionV3
      # setup model
      base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
      model = add_new_last_layer(base_model, nb_classes)
      
      filepath = "_{acc:.4f}_{loss:.4f}_{epoch:02d}_"
      checkpoint = ModelCheckpoint("models/{}inv3.model".format(filepath, monitor=['acc', 'loss'], verbose=1, save_best_only=True, mode='max'))

#  elif arch == 'dense201':
#      from keras.applications.densenet import DenseNet201
#      # setup model
#      base_model = DenseNet201(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
#      model = add_new_last_layer(base_model, nb_classes)
#      
#      filepath = "{acc:.4f}_{loss:.4f}_{epoch:02d}"
#      checkpoint = ModelCheckpoint("D:/CODE/New folder/{}dense.model".format(filepath, monitor=['acc', 'loss'], verbose=1, save_best_only=True, mode='max'))

  
  
  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=40,
      width_shift_range=0.3,
      height_shift_range=0.3,
      shear_range=0.3,
      zoom_range=0.3,
      horizontal_flip=True
  )
#  test_datagen = ImageDataGenerator(
#      preprocessing_function=preprocess_input,
#      rotation_range=45,
#      width_shift_range=0.4,
#      height_shift_range=0.4,
#      shear_range=0.4,
#      zoom_range=0.4,
#      horizontal_flip=True
#  )

  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BAT_SIZE,
  )

#  validation_generator = test_datagen.flow_from_directory(
#    args.val_dir,
#    target_size=(IM_WIDTH, IM_HEIGHT),
#    batch_size=batch_size,
#  )
  


  # transfer learning
  setup_to_transfer_learn(model, base_model)

    
  history_tl = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    #validation_data=validation_generator,
    #nb_val_samples=nb_val_samples,
    callbacks=[checkpoint],
    class_weight='auto')
  
  # fine-tuning
  setup_to_finetune(model)

  history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    #validation_data=validation_generator,
    #nb_val_samples=nb_val_samples,
    callbacks=[checkpoint],
    class_weight='auto')

  model.save(args.output_model_file)

  if args.plot:
    plot_training(history_ft)


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir", default="Task8/train_dir/")
  #a.add_argument("--val_dir", default="Task8/val_dir/")
  a.add_argument("--nb_epoch", default=100)
  a.add_argument("--batch_size", default=128)
  a.add_argument("--output_model_file", default="Res.model")
  a.add_argument("--plot", action="store_true")

  ArchList = ['inv3']

  args = a.parse_args()
#  if args.train_dir is None or args.val_dir is None:
#    a.print_help()
#    sys.exit(1)

  #if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
#    print("directories do not exist")
#    sys.exit(1)
for arch in ArchList:

  if arch == 'vgg19':
      IM_WIDTH, IM_HEIGHT = 224, 224 #fixed size for InceptionV3 which was 299 by 299
      NB_EPOCHS = 30
      BAT_SIZE = 16
      FC_SIZE = 1024
      NB_IV3_LAYERS_TO_FREEZE = 172
  elif arch == 'vgg16':
      IM_WIDTH, IM_HEIGHT = 224, 224 #fixed size for InceptionV3 which was 299 by 299
      NB_EPOCHS = 30
      BAT_SIZE = 16
      FC_SIZE = 1024
      NB_IV3_LAYERS_TO_FREEZE = 172
  elif arch == 'xception':
      IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3 which was 299 by 299
      NB_EPOCHS = 30
      BAT_SIZE = 16
      FC_SIZE = 1024
      NB_IV3_LAYERS_TO_FREEZE = 172
  elif arch == 'res50':
      IM_WIDTH, IM_HEIGHT = 224, 224 #fixed size for InceptionV3 which was 299 by 299
      NB_EPOCHS = 30
      BAT_SIZE = 16
      FC_SIZE = 1024
      NB_IV3_LAYERS_TO_FREEZE = 172
  elif arch == 'inv3':
      IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3 which was 299 by 299
      NB_EPOCHS = 30
      BAT_SIZE = 16
      FC_SIZE = 1024
      NB_IV3_LAYERS_TO_FREEZE = 172
  elif arch == 'dense201':
      IM_WIDTH, IM_HEIGHT = 224, 224 #fixed size for InceptionV3 which was 299 by 299
      NB_EPOCHS = 30
      BAT_SIZE = 16
      FC_SIZE = 1024
      NB_IV3_LAYERS_TO_FREEZE = 172
  
    
  train(args, arch)
  cuda.select_device(0)
  cuda.close()
  

  