import os
import pprint
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

cwd = os.getcwd()
folder_parent = os.path.join(cwd, 'data', "labelled", "split")

species = ["arm", "zea"]
locations = ["front", "rear"]
orientations = ["left", "right"]

# open all images following the naming convention

images = []
labels = []

for specie in species:
    for location in locations:
        for orientation in orientations:
            folder = os.path.join(folder_parent, specie, location, orientation)
            for file in os.listdir(folder):
                if file.endswith(".jpg"):
                    # print(os.path.join(folder, file))
                    img = cv2.imread(os.path.join(folder, file))
                    images.append(img)
                    labels.append(specie)

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (224, 224))
    return image, label

print("Loaded images and labels")

# convert to tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.map(process_images)

print(dataset)