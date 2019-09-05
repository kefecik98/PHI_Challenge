

import numpy as np

import cv2
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import pandas as pd

model = load_model("models/t7_0.6819_1.1256_96_iv3.model")
target_size = (299, 299) #fixed size for InceptionV3 architecture

xTest = np.load('Task7/X_test.npy')
yTest = np.zeros((len(xTest)))

for im in range(0, len(xTest)):
    imR1 = cv2.resize(xTest[im],dsize=target_size, interpolation=cv2.INTER_NEAREST)
    imR = np.expand_dims(imR1, axis=0)
    #imR = np.expand_dims(xTest[im], axis=0)
    imR = preprocess_input(imR.astype(np.float32))
    yTest[im] = np.argmax(model.predict(imR))
    print("Finished image --- "+ str(im)+"=="+str(yTest[im]))

df = pd.DataFrame(yTest)
df.to_csv("Task8/submission_t7_0.6819_1.1256_96_iv3.csv")
