#This is used to generate and save output of the encoder model after spliting the .h5, which is then used for quantization
import numpy as np
import tensorflow as tf
from SeismicSense_utils import generate_arrays_from_file,BidirectionalLSTM, Upsampling1DLayer, CustomCropping1D, f1
from tqdm import tqdm

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

x_val, y1_val,y2_val, y3_val, x_train, y1_train,y2_train, y3_train= load_data()
model = tf.keras.models.load_model('test_trainer_outputs/final_E_model.h5', custom_objects={'BidirectionalLSTM': BidirectionalLSTM})

batch=100
predictions=[]
data_generator = generate_arrays_from_file(x_train, batch) 
pbar_test = tqdm(total= int(np.ceil(len(x_train)/batch)))            
for _ in range(int(np.ceil(len(x_train) / batch))):
    pbar_test.update()
    new_data = next(data_generator)
    pred = model.predict(new_data, batch_size=batch)
    predictions.append(pred)

# Concatenate all the predictions into a single numpy array
predictions = np.concatenate(predictions)

# Save the predictions in a .npy file
np.save('/home/tza/STEAD/tza/E_output.npy', predictions)