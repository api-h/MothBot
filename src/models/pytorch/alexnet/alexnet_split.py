import os
import time
from tensorflow import keras, optimizers

cwd = os.getcwd()
folder_parent = os.path.join(cwd, 'data', "labelled", "split")

species = ["arm", "zea"]

train_images, val_images = keras.utils.image_dataset_from_directory(
    folder_parent,
    labels="inferred",
    label_mode="int",
    class_names=species,
    validation_split=0.1,
    seed=37,
    subset="both")

train_images, test_images = keras.utils.split_dataset(train_images,
                                                      left_size=8 / 9,
                                                      shuffle=True,
                                                      seed=37)

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96,
                        kernel_size=(11, 11),
                        strides=(4, 4),
                        activation='relu',
                        input_shape=(256, 256, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=256,
                        kernel_size=(5, 5),
                        strides=(1, 1),
                        activation='relu',
                        padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=384,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

root_logdir = os.path.join(os.curdir, "src\\models\\pytorch\\alexnet\\logs\\fit\\")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001),
              metrics=['accuracy'])
model.summary()

model.fit(train_images,
          epochs=50,
          validation_data=val_images,
          validation_freq=1,
          callbacks=[tensorboard_cb])

model.evaluate(test_images)