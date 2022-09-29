#!/usr/bin/env python3
##
#!pip install tensorflow --no-cache-dir

import sys
print("check\n\n\n\n\n\n", sys.version)

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print("tensorflow version", tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

## Define image properties:
imgDir = "data"

# ideal image sizes - not too small to discourage the traning but not too large to cause the code to
# run for too long
targetWidth, targetHeight = 200, 200
imageSize = (targetWidth, targetHeight)
channels = 1  # color channels

## define other constants, including command line argument defaults
epochs = 5
plot = False  # show plots?

## command line arguments
# check if this was run as a separate file (not inside notebook)
import __main__ as main
if hasattr(main, "__file__"):
    # run as file
    print("parsing command line arguments")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d",
                        help = "directory to read images from",
                        default = imgDir)
    parser.add_argument("--epochs", "-e",
                        help = "how many epochs",
                        default= epochs)
    parser.add_argument("--plot", "-p",
                        action = "store_true",
                        help = "plot a few wrong/correct results")
    args = parser.parse_args()
    imgDir = args.dir
    epochs = int(args.epochs)
    plot = args.plot
else:
    # run as notebook
    print("run interactively from", os.getcwd())
    imageDir = os.path.join(os.path.expanduser("~"),
                            "data", "images", "text", "language-text-images")
print("Load images from", imgDir)
print("epochs:", epochs)

## Prepare dataset for training model:
filenames = os.listdir(os.path.join(imgDir, "train"))
print(len(filenames), "images found")
trainingResults = pd.DataFrame({
    'filename':filenames,
    'category': pd.Series(filenames).str[-10:-8]
})
print("data files:")
print(trainingResults.sample(5))
nCategories = trainingResults.category.nunique()
print("categories:\n", trainingResults.category.value_counts())
## Create model
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,\
    MaxPooling2D, AveragePooling2D,\
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization

# sequential (not recursive) model (one input, one output)

model=Sequential()

# convolutional 2D layer with 16 layer filters to start with. I judged kernal size 3 and 
# strides to be 2 to be the best based on the layer filter value. Activation will be best
# with relu for almost all of the layers.
model.add(Conv2D(16,
              kernel_size=3,
              strides=2,
              activation='relu',
              kernel_initializer = initializers.HeNormal(),
              input_shape=(targetWidth, targetHeight, channels)))

# Normalization serving to speed up training and motivate higher learning rates
model.add(BatchNormalization())
# MaxPooling to help with over-fitting and reduce the computational cost.
model.add(MaxPooling2D(pool_size=2))

# Dropout layer to prevent over fitting and to have fair and more spread out training of data
model.add(Dropout(0.5))

# Same as the above convolutional layer but with 32 filters instead for a more in-depth
# learning. 
model.add(Conv2D(32,
              kernel_size=3,
              strides=2,
              activation='relu',
              kernel_initializer = initializers.HeNormal(),
              input_shape=(targetWidth, targetHeight, channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

# Dropout layer to prevent over fitting and to have fair and more spread out training of data
model.add(Dropout(0.5))

model.add(Conv2D(32,
              kernel_size=3,
              strides=2,
              activation='relu',
              kernel_initializer = initializers.HeNormal(),
              input_shape=(targetWidth, targetHeight, channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.25))

# Flattening of 2D image data for the dense layer to be trained with.
model.add(Flatten())

# Dense layer with 512 filter value to accurately learn the differences between languages. 
# Especially to prevent my code from training too less to recognize all languages to be the same.
model.add(Dense(512,
              kernel_initializer = initializers.HeNormal(),
              activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.25))

# Reduced filter value for a simpler parameters and to trim down the result into two distinct
# categories
model.add(Dense(32,
              kernel_initializer = initializers.HeNormal(),
              activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(nCategories,
              kernel_initializer = initializers.HeNormal(),
              activation='softmax'))

# chose optimizer Adam to speed up the traning process through accelerated gradient descent algorithm
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.summary()

## Training and validation data generator:
trainingGenerator = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    #horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
).\
    flow_from_dataframe(trainingResults,
                        os.path.join(imgDir, "train"),
                        x_col='filename', y_col='category',
                        target_size=imageSize,
                        class_mode='categorical',
                        color_mode="grayscale",
                        shuffle=True)
label_map = trainingGenerator.class_indices
## Model Training:
history = model.fit(
    trainingGenerator,
    epochs=epochs
)

## Validation data preparation:
validationDir = os.path.join(imgDir, "validation")
fNames = os.listdir(validationDir)
print(len(fNames), "validation images")
validationResults = pd.DataFrame({
    'filename': fNames,
    'category': pd.Series(fNames).str[-10:-8]
})
print(validationResults.shape[0], "validation files read from", validationDir)
validationGenerator = ImageDataGenerator(rescale=1./255).\
    flow_from_dataframe(validationResults,
                        os.path.join(imgDir, "validation"),
                        x_col='filename',
                        class_mode = None,
                        target_size = imageSize,
                        shuffle = False,
                        # do _not_ randomize the order!
                        # this would clash with the file name order!
                        color_mode="grayscale"
    )

## Make categorical prediction:
print(" --- Predicting on validation data ---")
phat = model.predict(validationGenerator)
print("Predicted probability array shape:", phat.shape)
print("Example:\n", phat[:5])

## Convert labels to categories:
validationResults['predicted'] = pd.Series(np.argmax(phat, axis=-1), index=validationResults.index)
print(validationResults.head())
labelMap = {v: k for k, v in label_map.items()}
validationResults["predicted"] = validationResults.predicted.replace(labelMap)
print("confusion matrix (validation)")
print(pd.crosstab(validationResults.category, validationResults.predicted))
print("Validation accuracy", np.mean(validationResults.category == validationResults.predicted))

## Print and plot misclassified results
wrongResults = validationResults[validationResults.predicted != validationResults.category]
rows = np.random.choice(wrongResults.index, min(4, wrongResults.shape[0]), replace=False)
print("Example wrong results (validation data)")
print(wrongResults.sample(min(10, wrongResults.shape[0])))
if plot:
    plt.figure(figsize=(12, 12))
    index = 1
    for row in rows:
        filename = wrongResults.loc[row, 'filename']
        predicted = wrongResults.loc[row, 'predicted']
        img = load_img(os.path.join(imgDir, "validation", filename), target_size=imageSize)
        plt.subplot(4, 2, index)
        plt.imshow(img)
        plt.xlabel(filename + " ({})".format(predicted))
        index += 1
    # now show correct results
    index = 5
    correctResults = validationResults[validationResults.predicted == validationResults.category]
    rows = np.random.choice(correctResults.index,
                            min(4, correctResults.shape[0]), replace=False)
    for row in rows:
        filename = correctResults.loc[row, 'filename']
        predicted = correctResults.loc[row, 'predicted']
        img = load_img(os.path.join(imgDir, "validation", filename), target_size=imageSize)
        plt.subplot(4, 2, index)
        plt.imshow(img)
        plt.xlabel(filename + " ({})".format(predicted))
        index += 1
    plt.tight_layout()
    plt.show()

## Training data predictions.
## Do these here to keep the in place for students
## 
print(" --- Predicting on training data: ---")
# do another generator: the same as training, just w/o shuffle
predictTrainGenerator = ImageDataGenerator(rescale=1./255).\
    flow_from_dataframe(trainingResults,
                        os.path.join(imgDir, "train"),
                        x_col='filename', y_col='category',
                        target_size=imageSize,
                        class_mode='categorical',
                        color_mode="grayscale",
                        shuffle=False  # do not shuffle!
    )
phat = model.predict(predictTrainGenerator)
trainingResults['predicted'] = pd.Series(np.argmax(phat, axis=-1), index=trainingResults.index)
trainingResults["predicted"] = trainingResults.predicted.replace(labelMap)
print("confusion matrix (training)")
print(pd.crosstab(trainingResults.category, trainingResults.predicted))
print("Train accuracy", np.mean(trainingResults.category == trainingResults.predicted))