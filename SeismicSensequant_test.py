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
from SeismicSense_utils import  generate_arrays_from_file, picker, _output_writter_test, _plotter
from tqdm import tqdm
import math
import random
import sys
import statistics



os.environ['CUDA_VISIBLE_DEVICES'] ='{}'.format(2)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


batch_size=10
input_hdf5= '/home/tza/STEAD/tza/merged.hdf5'
output_name='test_tester' 

args = {
    "input_hdf5": '/home/tza/STEAD/tza/merged.hdf5',
    "output_name": 'test_tester',
    "detection_threshold": 0.2,
    "P_threshold": 0.1,
    "S_threshold": 0.1,
    "number_of_plots": 5,
    "estimate_uncertainty": False,
    "number_of_sampling": 2,
    "loss_weights":   [0.05, 0.40, 0.55],
    "loss_types": ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
    "input_dimention": (6000,3),
    "batch_size": 100,
    }



save_dir = os.path.join(os.getcwd(), str(output_name)+'_outputs')
save_figs = os.path.join(save_dir, 'figures')

if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)  
os.makedirs(save_figs) 

test = np.load('/home/tza/STEAD/tza/x_EQdata_preprocessed_test.npy')
test_meta = np.load('/home/tza/STEAD/tza/EQtest.npy')

print('Writting results into: " ' + str(output_name)+'_outputs'+' "', flush=True)

start_training = time.time()          

csvTst = open(os.path.join(save_dir,'quantized_model.csv'), 'w')          
test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
test_writer.writerow(['network_code', 
                      'ID', 
                      'earthquake_distance_km', 
                      'snr_db', 
                      'trace_name', 
                      'trace_category', 
                      'trace_start_time', 
                      'source_magnitude', 
                      'p_arrival_sample',
                      'p_status', 
                      'p_weight',
                      's_arrival_sample', 
                      's_status', 
                      's_weight', 
                      'receiver_type',

                      'number_of_detections',
                      'detection_probability',
                      'detection_uncertainty',

                      'P_pick', 
                      'P_probability',
                      'P_uncertainty',
                      'P_error',

                      'S_pick',
                      'S_probability',
                      'S_uncertainty', 
                      'S_error'
                      ])  
csvTst.flush()        

plt_n = 0
print(test.shape)

data_generator = generate_arrays_from_file(test, batch_size) 
data_generator_meta = generate_arrays_from_file(test_meta, args['batch_size'])

pbar_test = tqdm(total= int(np.ceil(len(test)/batch_size))) 
for _ in range(int(np.ceil(len(test) / batch_size))):
    pbar_test.update()
    new_data_meta = next(data_generator_meta)
    new_data = next(data_generator)
    pred_DD_mean = np.zeros((0,6000,1)) 
    pred_PP_mean = np.zeros((0,6000,1)) 
    pred_SS_mean = np.zeros((0,6000,1)) 
    
    for counter in range(int(batch_size)):
        interpreter = tf.lite.Interpreter(model_path='keras_lstm/model_int8.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        test_sample = np.expand_dims(new_data[counter,:,:], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]['index'], test_sample)
        interpreter.invoke()
        pred_PP = interpreter.get_tensor(output_details[0]['index'])
        pred_DD = interpreter.get_tensor(output_details[1]['index'])
        pred_SS = interpreter.get_tensor(output_details[2]['index'])

        pred_DD_mean=np.concatenate((pred_DD_mean, pred_DD), axis=0)
        pred_PP_mean=np.concatenate((pred_PP_mean, pred_PP), axis=0)
        pred_SS_mean=np.concatenate((pred_SS_mean, pred_SS), axis=0)
    
    pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1]) 
    pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1]) 
    pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1]) 

    pred_DD_std = np.zeros((pred_DD_mean.shape))
    pred_PP_std = np.zeros((pred_PP_mean.shape))
    pred_SS_std = np.zeros((pred_SS_mean.shape))  
    

    test_set={}

    fl = h5py.File(input_hdf5, 'r')
    for ID in new_data_meta:
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get('data/'+str(ID))
        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get('data/'+str(ID))
        test_set.update( {str(ID) : dataset})                 

    for ts in range(pred_DD_mean.shape[0]): 
        evi =  new_data_meta[ts] 
        dataset = test_set[evi]  

        try:
            spt = int(dataset.attrs['p_arrival_sample']);
        except Exception:     
            spt = None

        try:
            sst = int(dataset.attrs['s_arrival_sample']);
        except Exception:     
            sst = None

        matches, pick_errors, yh3=picker(args, pred_DD_mean[ts], pred_PP_mean[ts], pred_SS_mean[ts],
                                               pred_DD_std[ts], pred_PP_std[ts], pred_SS_std[ts], spt, sst) 

        _output_writter_test(args,dataset, evi, test_writer, csvTst, matches, pick_errors)


        if plt_n < args['number_of_plots']:  

            _plotter(dataset,
                        evi,
                        args, 
                        save_figs, 
                        pred_DD_mean[ts], 
                        pred_PP_mean[ts],
                        pred_SS_mean[ts],
                        pred_DD_std[ts],
                        pred_PP_std[ts], 
                        pred_SS_std[ts],
                        matches)

        plt_n += 1

end_training = time.time()