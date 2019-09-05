
import numpy as np
from PIL import Image


xTrain = np.load('Task6/X_train.npy')
yTrain = np.load('Task6/y_train.npy')
xTest = np.load('Task6/X_test.npy')

for i in range(0,len(yTrain)):

    im = Image.fromarray(xTrain[i])
    if i <= 2108:
    	im.save("Task6/train_dir/{0}/{1}.jpeg".format(yTrain[i], i))
    else:
    	im.save("Task6/val_dir/{0}/{1}.jpeg".format(yTrain[i], i))
    print("train image completed ... " + str(i))

