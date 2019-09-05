

import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

import cv2
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import pandas as pd

model = load_model("C:/Users/Kadir/Desktop/PHI_Challenge/models/t8CNNiV3_45_0.638.model")
target_size = (229, 229) #fixed size for InceptionV3 architecture

xTest = np.load('Task8/X_test.npy')
yTest = np.zeros((len(xTest),2), dtype=int)

for im in range(0, len(xTest)):
    imR = cv2.resize(xTest[im],dsize=target_size, interpolation=cv2.INTER_NEAREST)
    imR = np.expand_dims(imR, axis=0)
    imR = preprocess_input(imR.astype(np.float32))
    yTest[1] = model.predict(imR)[0]
    yTest[0] = im
    print("Finished image --- "+ str(im))
    
df = pd.DataFrame(yTest)
df.to_csv("Task8/submission_t8CNNiV3_45_0.638.csv")

def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]


def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("cat", "dog")
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image", default="")
  a.add_argument("--image_url", help="url to image")
  a.add_argument("--model")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  model = load_model(args.model)
  if args.image is not None:
    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    plot_preds(img, preds)