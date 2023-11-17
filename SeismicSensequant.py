# converting the while .h5 model into tflite model
from __future__ import print_function 

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from sklearn import utils
import pathlib
import os
import shutil
import csv
import h5py
import time
import datetime
from tensorflow.keras.models import load_model
np.seterr(divide='ignore', invalid='ignore')
import math
from SeismicSense_utils import  generate_arrays_from_file, f1, CustomCropping1D, Upsampling1DLayer, BidirectionalLSTM, picker, _output_writter_test, _plotter

import random
import sys
import statistics
from sklearn.metrics import mean_absolute_error 
from tensorflow.keras.layers import Input


os.environ['CUDA_VISIBLE_DEVICES'] ='{}'.format(1)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)



def convert_model(model, tflite_modes):
    
    run_model = tf.function(lambda x: model(x))
    # Set the fixed input to the model as a concrete function
    # NEW HERE: I fix the bach size to 1, but keep the sequence size to None (dynamic)
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,6000,3], model.inputs[0].dtype))
    # save the Keras model with fixed input
    MODEL_DIR = "keras_lstm"
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    # Create converter from saved model
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    Choice="_float32"
    #tflite_modes==0 is for 32-bit float model..
    
    #code for compressed model..
    if(tflite_modes==1):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    #code for 8-bit quantized model..
    if(tflite_modes==2):
        Choice="_int8"
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
        def generate_representative_dataset():
            output=6000
            X_train=np.load('/home/tza/STEAD/tza/x_EQdata_train.npy')
            for i in range(int(X_train.shape[0]/100)):
                print(i,end="\r")
                yield [tf.expand_dims(X_train[i], axis=0)]
                #yield [X_train[i]]
        # Converter will use the above function to optimize quantization
        converter.representative_dataset = generate_representative_dataset
     
    #code for 8-bit quantized weights and 16-bit activation model..
    if(tflite_modes==3): 
        Choice="_int8w16"
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
     
    # Convert model
    tflite_model = converter.convert()
    open("keras_lstm/model"+Choice+".tflite", "wb").write(tflite_model)
    return tflite_model



model1 = load_model('./test_trainer_outputs/final_model.h5',  custom_objects={'BidirectionalLSTM':BidirectionalLSTM,'f1': f1, 'Upsampling1DLayer':Upsampling1DLayer,'CustomCropping1D':CustomCropping1D} )


tflite_model = convert_model(model1, 2)
