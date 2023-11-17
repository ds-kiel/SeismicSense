#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:44:14 2018
  
@author: mostafamousavi
""" 
   
    
from __future__ import print_function 

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow_model_optimization.sparsity import keras as sparsity
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
from SeismicSense_utils import  DataGenerator, lr_schedule ,f1,SeismicSense_original, CustomCropping1D, Upsampling1DLayer, BidirectionalLSTM
import argparse
from tqdm import tqdm
from SeismicSense_datagen import data_generation, data_generation_test
import math
import random
import sys
import statistics
from sklearn.metrics import mean_absolute_error 
from tensorflow.keras.layers import Input



np.set_printoptions(threshold=sys.maxsize)
 

parser = argparse.ArgumentParser(description='Inputs for SeismicSense')    
parser.add_argument("--mode", dest='mode', default='test', help="prepare, train,quant, test,test_accuracy")
parser.add_argument("--data_dir", dest='data_dir', default="/home/tza/STEAD/tza/merged.hdf5", type=str, help="Input file directory") 
parser.add_argument("--data_list", dest='data_list', default="/home/tza/STEAD/tza/merged.csv", type=str, help="Input csv file")
parser.add_argument("--input_model", dest='input_model', default="./test_trainer_outputs/final_model.h5", type=str, help="The pre-trained model used for the prediction")
parser.add_argument("--input_testdir", dest='input_testdir', default= "/home/tza/STEAD/tza/", type=str, help="List set set directory")
parser.add_argument("--batch_size", dest='batch_size', default= 100, type=int, help="batch size")  
parser.add_argument("--epochs", dest='epochs', default= 20, type=int, help="number of epochs (default: 100)")
parser.add_argument('--gpuid', dest='gpuid', type=int, default=2, help='specifyin the GPU')
parser.add_argument('--gpu_limit', dest='gpu_limit', type=float, default=0.8, help='limiting the GPU memory')
parser.add_argument("--input_dimention", dest='input_dimention', default=(6000, 3), type=int, help="a tuple including the time series lenght and number of channels.")  
parser.add_argument("--shuffle", dest='shuffle', default= True, type=bool, help="shuffling the list during the preprocessing and training")
parser.add_argument("--label_type",dest='label_type',  default='gaussian', type=str, help="label type for picks: 'gaussian', 'triangle', 'box' ") 
parser.add_argument("--normalization_mode", dest='normalization_mode', default='std', type=str, help="normalization mode for preprocessing: 'std' or 'max' ") 
parser.add_argument("--augmentation", dest='augmentation', default= True, type=bool, help="if True, half of each batch will be augmented")  
parser.add_argument("--add_event_r", dest='add_event_r', default= 0.6, type=float, help=" chance for randomly adding a second event into the waveform") 
parser.add_argument("--shift_event_r", dest='shift_event_r', default= 0.9, type=float, help=" shift the event") 
parser.add_argument("--add_noise_r", dest='add_noise_r', default= 0.5, type=float, help=" chance for randomly adding Gaussian noise into the waveform")  
parser.add_argument("--scale_amplitude_r", dest='scale_amplitude_r', default= None, type=float, help=" chance for randomly amplifying the waveform amplitude ") 
parser.add_argument("--pre_emphasis", dest='pre_emphasis', default= False, type= bool, help=" if raw waveform needs to be pre emphesis ")
parser.add_argument("--drop_channel_r", dest='drop_channel_r', default= 0.5, type= bool, help=" drop the channels")
parser.add_argument("--add_gap_r", dest='add_gap_r', default= 0.2, type= bool, help=" adding the gap")
parser.add_argument("--coda_ratio", dest='coda_ratio', default= 0.4, type= bool, help=" code ratio")
parser.add_argument("--train_valid_test_split", dest='train_valid_test_split', default=[0.85, 0.05, 0.10], type= float, help=" precentage for spliting the data into training, validation, and test sets")  
parser.add_argument("--patience", dest='patience', default= 5, type= int, help=" patience for early stop monitoring ") 
parser.add_argument("--detection_threshold",dest='detection_threshold',  default=0.20, type=float, help="Probability threshold for P pick")
parser.add_argument("--report", dest='report', default=False, type=bool, help="summary of the training settings and results.")
parser.add_argument("--plot_num", dest='plot_num', default= 50, type=int, help="number of plots for the test or predition results")
parser.add_argument("--tflite_modes", dest='tflite_modes', default= 0, type=int, help="flavor of tensorflow lite model")
parser.add_argument("--quant", dest='quant', default= 0, type=int, help="do you want to test the quantized model?")
parser.add_argument("--output", dest='output', default= 76, type=int, help="number of predictions")


 

args = parser.parse_args()
 

tflite=args.quant
tflite_modes=args.tflite_modes  

print("mode is: ",tflite_modes)
print('Mode is: {}'.format(args.mode))  


#Using GPUs
if (args.gpuid!=None ) :           
    os.environ['CUDA_VISIBLE_DEVICES'] ='{}'.format(args.gpuid)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)



##################################### Preparing Data ##########################################
###############################################################################################


    
if args.mode == 'prepare':               
    
    params = {
            'file_name': '/home/tza/STEAD/tza/merged.hdf5',
        
        
            'dim': 6000,
            'batch_size': 100,
            'n_channels': 3,
            'norm_mode': 'std',
            'augmentation': True,
            'add_event_r': 0.6,
            'add_gap_r':0.2,
            'shift_event_r': 0.9,  
            'add_noise_r': 0.5, 
            'scale_amplitude_r': None,
            'shuffle': True,
            'pre_emphasis': False,
            'drop_rate':0.2,
            'label_type':'gaussian',
            'shuffle':True, 
            'drop_channel_r':0.5,
            'coda_ratio':0.4,
            }  
    
    train_valid_test_split=[0.60, 0.20, 0.20]
    input_csv= str('/home/tza/STEAD/tza/merged.csv')
    df = pd.read_csv(input_csv)
    ev_list = df.trace_name.tolist()    
    np.random.shuffle(ev_list)     
    training = ev_list[:int(train_valid_test_split[0]*len(ev_list))]
    validation =  ev_list[int(train_valid_test_split[0]*len(ev_list)):
                            int(train_valid_test_split[0]*len(ev_list) + train_valid_test_split[1]*len(ev_list))]
    test =  ev_list[ int(train_valid_test_split[0]*len(ev_list) + train_valid_test_split[1]*len(ev_list)):]
    
    params_test = {'file_name':'/home/tza/STEAD/tza/merged.hdf5', 
                   'dim': 6000,
                   'batch_size': 100,
                   'n_channels': 3,
                   'norm_mode': 'std'} 
    np.save('/home/tza/STEAD/tza/EQtest', test) 
    data_generation_test(params_test, list_IDs=test)
    data_generation(params,list_IDs=training,tag='train')
    data_generation(params,list_IDs=validation,tag='val')
    

def load_data():
    y1_vdummy = np.load('/home/tza/STEAD/tza/y_EQlabelD_val.npy')
    y2_vdummy = np.load('/home/tza/STEAD/tza/y_EQlabelP_val.npy')
    y3_vdummy = np.load('/home/tza/STEAD/tza/y_EQlabelS_val.npy')
    X_vdummy = np.load('/home/tza/STEAD/tza/x_EQdata_val.npy')
    
    y1_dummy = np.load('/home/tza/STEAD/tza/y_EQlabelD_train.npy')
    y2_dummy = np.load('/home/tza/STEAD/tza/y_EQlabelP_train.npy')
    y3_dummy = np.load('/home/tza/STEAD/tza/y_EQlabelS_train.npy')
    X_dummy = np.load('/home/tza/STEAD/tza/x_EQdata_train.npy')
    return (X_vdummy, y1_vdummy, y2_vdummy, y3_vdummy, X_dummy, y1_dummy, y2_dummy, y3_dummy)




def trainer(output_name=None,                
            input_dimention=(6000, 3),
            cnn_blocks=1,
            lstm_blocks=2,
            padding='same',
            activation = 'relu',            
            drop_rate=0.1,         
            loss_weights=[0.05, 0.40, 0.55],
            loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            train_valid_test_split=[0.85, 0.05, 0.10],
            batch_size=200,
            epochs=200, 
            monitor='val_loss',
            patience=12,
            multi_gpu=False,
            number_of_gpus=4,
            use_multiprocessing=True):

    args = {
    "output_name": output_name,
    "input_dimention": input_dimention,
    "cnn_blocks": cnn_blocks,
    "lstm_blocks": lstm_blocks,
    "padding": padding,
    "activation": activation,
    "drop_rate": drop_rate,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "train_valid_test_split": train_valid_test_split,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience,           
    "multi_gpu": multi_gpu,
    "number_of_gpus": number_of_gpus,           
    "use_multiprocessing": use_multiprocessing
    }
    
    def train(args):
        save_dir, save_models=_make_dir(args['output_name'])
        callbacks=_make_callback(args, save_models)
        model=_build_model(args)
            
        start_training = time.time()
        
        param={
        'batch_size':args['batch_size'],
        'shuffle': False
        }
        
        
        x_val, y1_val,y2_val, y3_val, x_train, y1_train,y2_train, y3_train= load_data()
        training_generator = DataGenerator(x_train, y1_train, y2_train, y3_train,**param)
        val_generator = DataGenerator(x_val, y1_val, y2_val, y3_val,**param)
        history = model.fit(training_generator,
                        epochs=args['epochs'],
                        batch_size=args['batch_size'], 
                        callbacks=callbacks,
                        validation_data=val_generator)

        model.save(save_dir+'/final_model.h5')
        model.to_json()   
        model.save_weights(save_dir+'/model_weights.h5')
        end_training = time.time() 
        
    train(args)  
    
    
    
def _make_dir(output_name):
    
    """ 
    
    Make the output directories.
    Parameters
    ----------
    output_name: str
        Name of the output directory.
                   
    Returns
    -------   
    save_dir: str
        Full path to the output directory.
        
    save_models: str
        Full path to the model directory. 
        
    """   
    
    if output_name == None:
        print('Please specify output_name!') 
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name)+'_outputs')
        save_models = os.path.join(save_dir, 'models')      
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_models)
    return save_dir, save_models



def _build_model(args): 
    
    """ 
    
    Build and compile the model.
    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
               
    Returns
    -------   
    model: 
        Compiled model.
        
    """       
    
    inp = Input(shape=args['input_dimention'], name='input') 

    
    model=SeismicSense_original(
              padding=args['padding'],
              activationf =args['activation'], 
              loss_types=args['loss_types'],
              kernel_regularizer=keras.regularizers.l2(1e-6),
              bias_regularizer=keras.regularizers.l1(1e-4)  
               )(inp)
 
    model.summary()  
    return model  
    


def _split(args, save_dir):
    
    """ 
    
    Split the list of input data into training, validation, and test set.
    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_dir: str
       Path to the output directory. 
              
    Returns
    -------   
    training: str
        List of trace names for the training set. 
    validation : str
        List of trace names for the validation set. 
                
    """       
    
    df = pd.read_csv(args['input_csv'])
    ev_list = df.trace_name.tolist()    
    np.random.shuffle(ev_list)     
    training = ev_list[:int(args['train_valid_test_split'][0]*len(ev_list))]
    validation =  ev_list[int(args['train_valid_test_split'][0]*len(ev_list)):
                            int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list))]
    test =  ev_list[ int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list)):]
    np.save(save_dir+'/test', test)  
    return training, validation 



def _make_callback(args, save_models):
    
    """ 
    
    Generate the callback.
    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_models: str
       Path to the output directory for the models. 
              
    Returns
    -------   
    callbacks: obj
        List of callback objects. 
        
        
    """    
    
    m_name=str(args['output_name'])+'_{epoch:03d}.h5'   
    filepath=os.path.join(save_models, m_name)  
    early_stopping_monitor=EarlyStopping(monitor=args['monitor'], 
                                           patience=args['patience']) 
    checkpoint=ModelCheckpoint(filepath=filepath,
                                 monitor=args['monitor'], 
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True)  
    lr_scheduler=LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=args['patience']-2,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]
    return callbacks
 

def _document_training(history, model, start_training, end_training, save_dir, save_models, args): 

    """ 
    
    Write down the training results.
    Parameters
    ----------
    history: dic
        Training history.  
   
    model: 
        Trained model.  
    start_training: datetime
        Training start time. 
    end_training: datetime
        Training end time.    
         
    save_dir: str
        Path to the output directory. 
    save_models: str
        Path to the folder for saveing the models.  
      
     
    args: dic
        A dictionary containing all of the input parameters. 
              
    Returns
    -------- 
    ./output_name/history.npy: Training history.    
    ./output_name/X_report.txt: A summary of parameters used for the prediction and perfomance.
    ./output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.         
    ./output_name/X_learning_curve_loss.png: The learning curve of loss.  
        
        
    """   
    
    np.save(save_dir+'/history',history)
    model.save(save_dir+'/final_model.h5')
    model.to_json()   
    model.save_weights(save_dir+'/model_weights.h5')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['detector_loss'])
    ax.plot(history.history['picker_P_loss'])
    ax.plot(history.history['picker_S_loss'])
    try:
        ax.plot(history.history['val_loss'], '--')
        ax.plot(history.history['val_detector_loss'], '--')
        ax.plot(history.history['val_picker_P_loss'], '--')
        ax.plot(history.history['val_picker_S_loss'], '--') 
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss', 
               'val_loss', 'val_detector_loss', 'val_picker_P_loss', 'val_picker_S_loss'], loc='upper right')
    except Exception:
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss'], loc='upper right')  
        
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 
       
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['detector_f1'])
    ax.plot(history.history['picker_P_f1'])
    ax.plot(history.history['picker_S_f1'])
    try:
        ax.plot(history.history['val_detector_f1'], '--')
        ax.plot(history.history['val_picker_P_f1'], '--')
        ax.plot(history.history['val_picker_S_f1'], '--')
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1', 'val_detector_f1', 'val_picker_P_f1', 'val_picker_S_f1'], loc='lower right')
    except Exception:
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1'], loc='lower right')        
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_f1.png'))) 

    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta    
    
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n') 
        the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')  
        the_file.write('================== Model Parameters ========================='+'\n')   
        the_file.write('input_dimention: '+str(args['input_dimention'])+'\n')
        the_file.write('cnn_blocks: '+str(args['cnn_blocks'])+'\n')
        the_file.write('lstm_blocks: '+str(args['lstm_blocks'])+'\n')
        the_file.write('padding_type: '+str(args['padding'])+'\n')
        the_file.write('activation_type: '+str(args['activation'])+'\n')        
        the_file.write('drop_rate: '+str(args['drop_rate'])+'\n')            
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')    
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n') 
        the_file.write('================== Training Parameters ======================'+'\n')  
        the_file.write('mode of training: '+str(args['mode'])+'\n')   
        the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('epochs: '+str(args['epochs'])+'\n')   
        the_file.write('train_valid_test_split: '+str(args['train_valid_test_split'])+'\n')
        the_file.write('monitor: '+str(args['monitor'])+'\n')
        the_file.write('patience: '+str(args['patience'])+'\n') 
        the_file.write('multi_gpu: '+str(args['multi_gpu'])+'\n')
        the_file.write('number_of_gpus: '+str(args['number_of_gpus'])+'\n') 
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')  
        the_file.write('================== Training Performance ====================='+'\n')  
        the_file.write('finished the training in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds,2)))                         
        the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
        the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last detector_loss: '+str(history.history['detector_loss'][-1])+'\n')
        the_file.write('last picker_P_loss: '+str(history.history['picker_P_loss'][-1])+'\n')
        the_file.write('last picker_S_loss: '+str(history.history['picker_S_loss'][-1])+'\n')
        the_file.write('last detector_f1: '+str(history.history['detector_f1'][-1])+'\n')
        the_file.write('last picker_P_f1: '+str(history.history['picker_P_f1'][-1])+'\n')
        the_file.write('last picker_S_f1: '+str(history.history['picker_S_f1'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('label_type: '+str(args['label_type'])+'\n')
        the_file.write('augmentation: '+str(args['augmentation'])+'\n')
        the_file.write('shuffle: '+str(args['shuffle'])+'\n')               
        the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
        the_file.write('add_event_r: '+str(args['add_event_r'])+'\n')
        the_file.write('add_noise_r: '+str(args['add_noise_r'])+'\n')   
        the_file.write('shift_event_r: '+str(args['shift_event_r'])+'\n')                            
        the_file.write('drop_channel_r: '+str(args['drop_channel_r'])+'\n')            
        the_file.write('scale_amplitude_r: '+str(args['scale_amplitude_r'])+'\n')            
        the_file.write('pre_emphasis: '+str(args['pre_emphasis'])+'\n')            

        
def tflite_conversion(model):
    
    run_model = tf.function(lambda x: model(x))
    # Set the fixed input to the model as a concrete function
    # NEW HERE: I fix the bach size to 1, but keep the sequence size to None (dynamic)
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,6000,3], model.inputs[0].dtype))
    # save the Keras model with fixed input
    MODEL_DIR = "keras_lstm"
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    # Create converter from saved model
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    Choice="_int8"
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    converter.target_spec.supported_ops =[tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
    x_train = np.load('/home/tza/STEAD/tza/x_EQdata_train.npy')
    def generate_representative_dataset():
        for i in range(int(x_train.shape[0]/100)):
            print(i,end="\r")
            yield [tf.expand_dims(x_train[i], axis=0)]

    converter.representative_dataset = generate_representative_dataset

    tflite_model = converter.convert()
    open("keras_lstm/model"+Choice+".tflite", "wb").write(tflite_model)
    return tflite_model


if args.mode == 'train':   
    trainer(
        output_name='test_trainer',                
        cnn_blocks=0,
        lstm_blocks=1,
        padding='same',
        activation='relu',
        drop_rate=0.2,
        train_valid_test_split=[0.60, 0.20, 0.20],
        batch_size=100,
        epochs=5, 
        patience=5)

if args.mode == 'quant':    
    model1 = load_model(args.input_model, custom_objects={ 'CustomCropping1D':CustomCropping1D, 'Upsampling1DLayer':Upsampling1DLayer, 'BidirectionalLSTM': BidirectionalLSTM,'f1': f1 })
    tflite_model = tflite_conversion(model1)
    
  



    
