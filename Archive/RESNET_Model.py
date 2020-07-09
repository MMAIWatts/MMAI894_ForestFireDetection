#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.applications import vgg16
import matplotlib.pyplot as plt
import time
from PIL import Image
import pandas as pd
import model_evaluation_utils as meu


# In[2]:


def showNumpyImage(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()


# In[3]:


# Load Data
data = np.load("training_data.npy", allow_pickle = True)

# Split into labels and images then test and train
x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Scale images
train_x_scaled = train_x.astype('float32')
test_x_scaled = test_x.astype('float32')
train_x_scaled /= 255
test_x_scaled /= 255


# In[7]:




# Configure base model
input_shape = (224, 224, 3)
model_vgg16 = vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)
output = model_vgg16.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(model_vgg16.input, output)

# Set blocks 4 and 5 to be fine tuneable
vgg_model.trainable = True
set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])


# In[8]:


# Run model for unaugmented data
from keras import optimizers
model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
              
history = model.fit(train_x_scaled, train_y, batch_size=32, epochs=10, 
                              verbose=1)


# In[ ]:


test_predictions = model.predict(test_x_scaled)
test_predictions_labelled = [0 if x<0.1 else 1 for x in test_predictions]


# In[ ]:


# Display performance metrics
meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=test_predictions_labelled, classes=list(set(test_y)))


# In[ ]:


# Extract just fire images for augmentation
# TODO: this is not very efficient
data_df = pd.DataFrame(data)
just_fire = data_df[data_df[1] ==1]
just_fire_images = just_fire[0].tolist()
just_fire_labels = just_fire[1].tolist()


# In[ ]:


fire_data_aug = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, vertical_flip=True, fill_mode='nearest')


# In[ ]:


# Produce 12 random augmented image per fire image
from numpy import expand_dims
aug_images = []
aug_image_lables = []
for image in just_fire_images[0]:
    image = expand_dims(image,0)
    it = fire_data_aug.flow(image, batch_size=1)
    for i in range(12):
        batch = it.next()
        aug_images.append(batch[0])
        aug_image_lables.append(1)


# In[ ]:


# Create new augmented training and label set by combining original training and augmented training sets
train_x_aug = np.concatenate((train_x_scaled,np.array(aug_images)))
train_y_aug = np.concatenate((train_y,np.array(aug_image_lables)))


# In[ ]:




# Run model with augmented data 
model_aug = Sequential()
model_aug.add(vgg_model)
model_aug.add(Dense(512, activation='relu', input_dim=input_shape))
model_aug.add(Dropout(0.3))
model_aug.add(Dense(512, activation='relu'))
model_aug.add(Dropout(0.3))
model_aug.add(Dense(1, activation='sigmoid'))

model_aug.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

model_aug.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
              
history_aug = model_aug.fit(train_x_aug, train_y_aug, batch_size=32, epochs=10, 
                              verbose=1)


# In[ ]:


# Make preditions with augmented data model & convert to binary
test_predictions_aug = model_aug.predict(test_x_scaled)
test_predictions_aug_labelled = [0 if x<0.1 else 1 for x in test_predictions_aug]


# In[ ]:


# Display performance metrics (9X augmentation, 3 Epoch)
meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=test_predictions_aug_labelled, classes=list(set(test_y)))


# In[ ]:


# Display performance metrics (9X augmentation, 10 Epoch)
meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=test_predictions_aug_labelled, classes=list(set(test_y)))


# In[ ]:




# Display performance metrics (12X augmentation, incl. vertical_flip, 10 Epoch)
meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=test_predictions_aug_labelled, classes=list(set(test_y)))

