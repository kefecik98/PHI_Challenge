
# coding: utf-8

# In[1]:


from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


import numpy as np


# In[2]:


from keras import regularizers
from keras.models import load_model
from keras.models import Model
#import matplotlib
#from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import backend as K
from keras import applications
#from helper import getclassnames, get_train_data, get_test_data, plot_images
#from helper import plot_model, predict_classes, visualize_errors
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
import numpy as np
import pandas as pd
import cv2

# In[3]:


img_width, img_height = 299, 299


from keras.applications.inception_v3 import preprocess_input

xTest = np.load('Task7/X_test.npy')
input_shape = (img_width, img_height, 3)

model_input = Input(shape=input_shape)



from keras.layers import Average




def ensemble(models, model_input):
    
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)
    
    model = Model(inputs=model_input, outputs=y, name='ensemble')
    
    return model


from keras.models import load_model


model_mbn2 = load_model("models/t6_0.7002_1.0559_93_mbn2.model")
model_iv3_1 = load_model("models/t6_0.7396_0.9810_84_iv3.model")
model_iv3_2 = load_model("models/t6_0.7287_0.9836_59_iv3.model")

model_mbn2.name = 'model_mbn2'
model_iv3_1.name = 'model_iv3_1'
model_iv3_2.name = 'model_iv3_2'

models  =  [model_mbn2, model_iv3_1, model_iv3_2]
ensemble_model = ensemble(models, model_input)
ensemble_model.summary()


# In[ ]:
target_size = (299, 299)
yTest = np.zeros((len(xTest)))
for im in range(0, len(xTest)):
    imR1 = cv2.resize(xTest[im],dsize=target_size, interpolation=cv2.INTER_NEAREST)
    imR = np.expand_dims(imR1, axis=0)
    #imR = np.expand_dims(xTest[im], axis=0)
    imR = preprocess_input(imR.astype(np.float32))
    yTest[im] = np.argmax(ensemble_model.predict(imR))
    print("Finished image --- "+ str(im)+"=="+str(yTest[im]))

df = pd.DataFrame(yTest)
df.to_csv("Task8/submission_t6_en.csv")



