

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import recall_m, precision_m, f1_m
x_train = np.load('./results/x_train.npy')
x_valid = np.load('./results/x_valid.npy')
y_train = np.load('./results/y_train.npy')
y_valid = np.load('./results/y_valid.npy')
x_train =  x_train[:,:,:,0:6]
x_valid = x_valid[:,:,:,0:6]
y_train = y_train[:,:,:,0:6]
y_valid = y_valid[:,:,:,0:6]
 
class MiConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']  # Elimina el argumento no deseado
        super().__init__(*args, **kwargs)

model = tf.keras.models.load_model('./results/best_model.h5', custom_objects={'Conv2DTranspose': MiConv2DTranspose})


threshold = 0.5
pred_img = model.predict(x_valid)
pred_img = (pred_img > threshold).astype(np.uint8)
np.save('./results/pred_img.npy', pred_img)

