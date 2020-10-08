from django.shortcuts import render
from django.http import HttpResponse

# In [0]:
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#In [2]:
df = pd.DataFrame()

#In [3]:
imgs = []

#In [4]:
labels = []

#In [5]:
directory = "C:\\Users\\harsh\\OneDrive\\Documents\\CFRCE\\AI\\LeafSaver\\Negative\\"

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        fullpath = os.path.join(directory, filename)
        fullpath_without_extension = os.path.join(directory, filename.split('.')[0])
        img = Image.open(fullpath)
        img = img.resize((100, 100), Image.ANTIALIAS)
        imgs.append(np.asarray(img))
        labels.append(0)
        continue
    else:
        continue

#In [6]:
directory = "C:\\Users\\harsh\\OneDrive\\Documents\\CFRCE\\AI\\LeafSaver\\Positive\\"

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        fullpath = os.path.join(directory, filename)
        fullpath_without_extension = os.path.join(directory, filename.split('.')[0])
        img = Image.open(fullpath)
        img = img.resize((100, 100), Image.ANTIALIAS)
        imgs.append(np.asarray(img))
        labels.append(1)
        continue
    else:
        continue

#In [7]:
df['img_array'] = imgs
df['infected'] = labels

#In [8]:
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100, 3)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(2, activation="softmax")
])

#In [9]:
train_images = []

#In [10]:
for img in df['img_array']:
    train_images.append(img)

#In [11]:
train_images = np.asarray(train_images)

#In [12]:
train_labels = df['infected']
train_labels = np.asarray(train_labels)

#In [13]:
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=70)

#In [14]:
label = ['Not infected', 'Infected']

#In [15]:
# fullpath = "C:\\Users\\harsh\\OneDrive\\Documents\\CFRCE\\AI\\LeafSaver\\test\\testimage.png"
fullpath = "C:\\xampp\\htdocs\\leaf_saver\\uploads\\esp32-cam.jpg"
img = Image.open(fullpath)
img = img.resize((100, 100), Image.ANTIALIAS)
test = []
test.append(np.asarray(img))
test = np.asarray(test)
prediction = model.predict(test)

# Create your views here.
def homePageView(request):
    return HttpResponse(label[np.argmax(prediction)])