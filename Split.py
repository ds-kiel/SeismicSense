#I use this file to split my .h5 model into 4 submodels
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv1D, Input, BatchNormalization
from SeismicSense_utils import f1, Upsampling1DLayer, CustomCropping1D, BidirectionalLSTM
import numpy as np
from tensorflow.keras.optimizers import Adam



model = tf.keras.models.load_model('test_trainer_outputs/final_model.h5', custom_objects={'f1': f1, 'Upsampling1DLayer':Upsampling1DLayer, 'BidirectionalLSTM': BidirectionalLSTM, 'CustomCropping1D': CustomCropping1D})



inputs = model.input
outputs = model.get_layer('encoder_block').output
model1 = tf.keras.Model(inputs=inputs, outputs=outputs)

encoder_output = model.get_layer('encoder_block').output
detector_output = model.get_layer('detector').output
model2 = tf.keras.Model(inputs=encoder_output, outputs=detector_output)

encoder_output = model.get_layer('encoder_block').output
pickerp_output = model.get_layer('picker_P').output
model3 = tf.keras.Model(inputs=encoder_output, outputs=pickerp_output)

encoder_output = model.get_layer('encoder_block').output
pickers_output = model.get_layer('picker_S').output
model4 = tf.keras.Model(inputs=encoder_output, outputs=pickers_output)


model1.compile(optimizer='adam')
model2.compile(loss='binary_crossentropy', optimizer='adam',  metrics=[f1])
model3.compile(loss='binary_crossentropy', optimizer='adam',  metrics=[f1])
model4.compile(loss='binary_crossentropy', optimizer='adam',  metrics=[f1])


model1.save('test_trainer_outputs/final_E_model.h5')
model2.save('test_trainer_outputs/final_D_model.h5')
model3.save('test_trainer_outputs/final_P_model.h5')
model4.save('test_trainer_outputs/final_S_model.h5')


test = np.load('/home/tza/STEAD/tza/x_EQdata_preprocessed_test.npy')
reshaped_input_data = np.reshape(test[2], (1, 6000, 3))
output= model1.predict(reshaped_input_data)
outputD = model2.predict(output)
outputP = model3.predict(output)
outputS = model4.predict(output)
