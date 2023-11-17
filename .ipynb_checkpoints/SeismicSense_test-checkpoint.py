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
from SeismicSense_utils import  generate_arrays_from_file, f1,CustomCropping1D, Upsampling1DLayer, BidirectionalLSTM, picker, _output_writter_test, _plotter
from tqdm import tqdm
import math
import random
import sys
import statistics

os.environ['CUDA_VISIBLE_DEVICES'] ='{}'.format(3)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def tester(input_hdf5=None,
           input_model=None,
           output_name=None,
           detection_threshold=0.20,                
           P_threshold=0.1,
           S_threshold=0.1, 
           number_of_plots=100,
           estimate_uncertainty=True, 
           number_of_sampling=5,
           loss_weights=[0.05, 0.40, 0.55],
           gpuid=0,
           loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
           input_dimention=(6000, 3),
           batch_size=500):
   
    args = {
    "input_hdf5": input_hdf5,
    "input_model": input_model,
    "output_name": output_name,
    "detection_threshold": detection_threshold,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "number_of_plots": number_of_plots,
    "estimate_uncertainty": estimate_uncertainty,
    "number_of_sampling": number_of_sampling,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "input_dimention": input_dimention,
    "batch_size": batch_size,
    "gpuid": gpuid,
    }  

    save_dir = os.path.join(os.getcwd(), str(args['output_name'])+'_outputs')
    save_figs = os.path.join(save_dir, 'figures')
 
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)  
    os.makedirs(save_figs) 
 
    test_meta = np.load('/home/tza/STEAD/tza/EQtest.npy')
    test = np.load('/home/tza/STEAD/tza/x_EQdata_preprocessed_test.npy')
    
    print('Loading the model ...', flush=True)        
    model = load_model(args['input_model'], custom_objects={ 'CustomCropping1D':CustomCropping1D, 'Upsampling1DLayer':Upsampling1DLayer,'BidirectionalLSTM': BidirectionalLSTM,'f1': f1 })
                
    model.compile(loss = args['loss_types'],
                  loss_weights =  args['loss_weights'],           
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
    
    print('Loading is complete!', flush=True)  
    print('Testing ...', flush=True)    
    print('Writting results into: " ' + str(args['output_name'])+'_outputs'+' "', flush=True)
    
    start_training = time.time()          

    csvTst = open(os.path.join(save_dir,'X_test_results.csv'), 'w')          
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
    data_generator = generate_arrays_from_file(test, args['batch_size']) 
    data_generator_meta = generate_arrays_from_file(test_meta, args['batch_size'])
    
    pbar_test = tqdm(total= int(np.ceil(len(test)/args['batch_size'])))            
    for _ in range(int(np.ceil(len(test) / args['batch_size']))):
        pbar_test.update()
        new_data = next(data_generator)
        new_data_meta = next(data_generator_meta)

        if args['estimate_uncertainty']:
            pred_DD = []
            pred_PP = []
            pred_SS = []          
            for mc in range(args['number_of_sampling']):
                predD, predP, predS = model.predict(new_data, batch_size=args['batch_size'])
                pred_DD.append(predD)
                pred_PP.append(predP)               
                pred_SS.append(predS)

            pred_DD = np.array(pred_DD).reshape(args['number_of_sampling'], len(new_data), args['input_dimention'][0])
            pred_DD_mean = pred_DD.mean(axis=0)
            pred_DD_std = pred_DD.std(axis=0)  

            pred_PP = np.array(pred_PP).reshape(args['number_of_sampling'], len(new_data), args['input_dimention'][0])
            pred_PP_mean = pred_PP.mean(axis=0)
            pred_PP_std = pred_PP.std(axis=0)      

            pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(new_data), args['input_dimention'][0])
            pred_SS_mean = pred_SS.mean(axis=0)
            pred_SS_std = pred_SS.std(axis=0) 

        else:          
            pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict_generator(generator=test_generator)
            pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1]) 
            pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1]) 
            pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1]) 

            pred_DD_std = np.zeros((pred_DD_mean.shape))
            pred_PP_std = np.zeros((pred_PP_mean.shape))
            pred_SS_std = np.zeros((pred_SS_mean.shape))           

        test_set={}
        
        fl = h5py.File(args['input_hdf5'], 'r')
        
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


            if plt_n < 50:  

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
    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta     
                    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')         
        the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')      
        the_file.write('input_model: '+str(args['input_model'])+'\n')
        the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')  
        the_file.write('================== Testing Parameters ======================='+'\n')    
        the_file.write('finished the test in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2))) 
        the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('total number of tests '+str(len(test))+'\n')
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('estimate uncertainty: '+str(args['estimate_uncertainty'])+'\n')
        the_file.write('number of Monte Carlo sampling: '+str(args['number_of_sampling'])+'\n')             
        the_file.write('detection_threshold: '+str(args['detection_threshold'])+'\n')            
        the_file.write('P_threshold: '+str(args['P_threshold'])+'\n')
        the_file.write('S_threshold: '+str(args['S_threshold'])+'\n')
        the_file.write('number_of_plots: '+str(args['number_of_plots'])+'\n')                        

        
        
tester(input_hdf5='/home/tza/STEAD/tza/merged.hdf5',
       input_model='/home/tza/Seismicsense/test_trainer_outputs/final_model.h5',
       output_name='test_tester',
       detection_threshold=0.20,                
       P_threshold=0.1,
       S_threshold=0.1, 
       number_of_plots=10,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       batch_size=10,
       )