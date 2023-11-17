import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import signal


import h5py


#******************************Training and validation data preprocessing**************************************************

 

def normalize( data, mode = 'max'):  
    'Normalize waveforms in each batch'

    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data

def scale_amplitude( data, rate):
    'Scale amplitude or waveforms'

    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data *= np.random.uniform(1, 3)
    elif tmp < 2*rate:
        data /= np.random.uniform(1, 3)
    return data

def drop_channel( data, snr, rate):
    'Randomly replace values of one or two components to zeros in earthquake data'

    data = np.copy(data)
    if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
        c1 = np.random.choice([0, 1])
        c2 = np.random.choice([0, 1])
        c3 = np.random.choice([0, 1])
        if c1 + c2 + c3 > 0:
            data[..., np.array([c1, c2, c3]) == 0] = 0
    return data

def drop_channel_noise( data, rate):
    'Randomly replace values of one or two components to zeros in noise data'

    data = np.copy(data)
    if np.random.uniform(0, 1) < rate: 
        c1 = np.random.choice([0, 1])
        c2 = np.random.choice([0, 1])
        c3 = np.random.choice([0, 1])
        if c1 + c2 + c3 > 0:
            data[..., np.array([c1, c2, c3]) == 0] = 0
    return data

def add_gaps( data, rate): 
    'Randomly add gaps (zeros) of different sizes into waveforms'

    data = np.copy(data)
    gap_start = np.random.randint(0, 4000)
    gap_end = np.random.randint(gap_start, 5500)
    if np.random.uniform(0, 1) < rate: 
        data[gap_start:gap_end,:] = 0           
    return data  

def add_noise( data, snr, rate):
    'Randomly add Gaussian noie with a random SNR into waveforms'

    data_noisy = np.empty((data.shape))
    if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
        data_noisy = np.empty((data.shape))
        data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
        data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
        data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
    else:
        data_noisy = data
    return data_noisy   

def adjust_amplitude_for_multichannels( data):
    'Adjust the amplitude of multichaneel data'

    tmp = np.max(np.abs(data), axis=0, keepdims=True)
    assert(tmp.shape[-1] == data.shape[-1])
    if np.count_nonzero(tmp) > 0:
      data *= data.shape[-1] / np.count_nonzero(tmp)
    return data

def label( a=0, b=20, c=40):  
    'Used for triangolar labeling'

    z = np.linspace(a, c, num = 2*(b-a)+1)
    y = np.zeros(z.shape)
    y[z <= a] = 0
    y[z >= c] = 0
    first_half = np.logical_and(a < z, z <= b)
    y[first_half] = (z[first_half]-a) / (b-a)
    second_half = np.logical_and(b < z, z < c)
    y[second_half] = (c-z[second_half]) / (c-b)
    return y

def add_event( data, addp, adds, coda_end, snr, rate): 
    'Add a scaled version of the event into the empty part of the trace'

    added = np.copy(data)
    additions = None
    spt_secondEV = None
    sst_secondEV = None
    if addp and adds:
        s_p = adds - addp
        if np.random.uniform(0, 1) < rate and all(snr>=10.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
            secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
            scaleAM = 1/np.random.randint(1, 10)
            space = data.shape[0]-secondEV_strt  
            added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
            added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
            added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
            spt_secondEV = secondEV_strt   
            if  spt_secondEV + s_p + 21 <= data.shape[0]:
                sst_secondEV = spt_secondEV + s_p
            if spt_secondEV and sst_secondEV:                                                                     
                additions = [spt_secondEV, sst_secondEV] 
                data = added

    return data, additions    


def shift_event( data, addp, adds, coda_end, snr, rate): 
    'Randomly rotate the array to shift the event location'

    org_len = len(data)
    data2 = np.copy(data)
    addp2 = adds2 = coda_end2 = None;
    if np.random.uniform(0, 1) < rate:             
        nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
        data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
        data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
        data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]

        if addp+nrotate >= 0 and addp+nrotate < org_len:
            addp2 = addp+nrotate;
        else:
            addp2 = None;
        if adds+nrotate >= 0 and adds+nrotate < org_len:               
            adds2 = adds+nrotate;
        else:
            adds2 = None;                   
        if coda_end+nrotate < org_len:                              
            coda_end2 = coda_end+nrotate 
        else:
            coda_end2 = org_len                 
        if addp2 and adds2:
            data = data2;
            addp = addp2;
            adds = adds2;
            coda_end= coda_end2;                                      
    return data, addp, adds, coda_end      

def pre_emphasis( params, data, pre_emphasis=0.97):
    'apply the pre_emphasis'

    for ch in range(params['n_channels']): 
        bpf = data[:, ch]  
        data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
    return data

# Define a custom key function to extract the event number
def get_event_number(event_index):
    return int(event_index.replace("Event_", ""))
def data_generation(params, list_IDs,tag):
    'read the waveforms' 
    lengthT=len(list_IDs)  
    X = np.zeros((lengthT, params['dim'], params['n_channels']))
    y1 = np.zeros((lengthT, params['dim'], 1))
    y2 = np.zeros((lengthT, params['dim'], 1))
    y3 = np.zeros((lengthT, params['dim'], 1))
    fl = h5py.File(params['file_name'], 'r')

    # Generate data
    csv_data = pd.read_csv('/home/tza/STEAD/tza/Testdata2/filtered_metadata-4.csv')

    
    '''with h5py.File("waveforms-4.hdf5", "r") as hdf5_file:
    # Get the group names and sort them numerically
    group_names = list(hdf5_file.keys())
    group_names.sort(key=lambda x: int(x.split("_")[1]))
    
    # Iterate through the sorted group names
    for group_name in group_names:
        group = hdf5_file[group_name]
        
        # Iterate through the datasets (Channel_j) in the group
        for dataset_name in group.keys():
            dataset = group[dataset_name]
            
            # Access the waveform data
            waveform_data = dataset[:]'''

    
    


    dataset = []
    with h5py.File(params['file_name'], 'r') as hdf5_file:
        # Get the list of event indices and sort them
        event_indices = list(hdf5_file.keys())
        sorted_event_indices = sorted(event_indices, key=get_event_number)

        for event_index in sorted_event_indices:
            index_number = get_event_number(event_index)

            if index_number in list_IDs:
                event_group = hdf5_file[event_index]
                event_waveforms = []
                for channel_index in event_group:
                    channel_data = np.array(event_group[channel_index])
                    event_waveforms.append(channel_data)
                dataset.append(event_waveforms)

    # Convert waveforms to a NumPy array
    dataset = np.array(dataset)
    dataset = np.transpose(dataset, (0, 2, 1))

    
    i=0
    for ID in enumerate(list_IDs):
        spt =int(csv_data['trace_P_arrival_sample'][ID[1]])    
        sst =int(csv_data['trace_S_arrival_sample'][ID[1]])
        #if(tag=="train"):
            #print(str(spt)+"****"+str(sst))
            #print(csv_data['station_code'][ID[1]])
            
        if params['norm_mode']:                    
            data = normalize(dataset[i], params['norm_mode'])                          

        X[i, :, :] = data                                       

        ## labeling 
        if params['label_type']  == 'gaussian': 
            sd = None    
            if spt and sst:
                sd = sst - spt  

            if sd and sst:
                if sst+int(0.4*sd) <= params['dim']: 
                    y1[i, spt:int(sst+(0.4*sd)), 0] = 1 
                else:
                    y1[i, spt:params['dim'], 0] = 1                       

            if spt and (spt-20 >= 0) and (spt+20 < params['dim']):
                y2[i, spt-20:spt+20, 0] = np.exp(-(np.arange(spt-20,spt+20)-spt)**2/(2*(10)**2))[:params['dim']-(spt-20)]                
            elif spt and (spt-20 < params['dim']):
                y2[i, 0:spt+20, 0] = np.exp(-(np.arange(0,spt+20)-spt)**2/(2*(10)**2))[:params['dim']-(spt-20)]

            if sst and (sst-20 >= 0) and (sst-20 < params['dim']):
                y3[i, sst-20:sst+20, 0] = np.exp(-(np.arange(sst-20,sst+20)-sst)**2/(2*(10)**2))[:params['dim']-(sst-20)]
            elif sst and (sst-20 < params['dim']):
                y3[i, 0:sst+20, 0] = np.exp(-(np.arange(0,sst+20)-sst)**2/(2*(10)**2))[:params['dim']-(sst-20)]
        i=i+1

    fl.close()
    X=X.astype('float32')
    y1=y1.astype('float32')
    y2=y2.astype('float32')
    y3=y3.astype('float32')
                                 
                                                                    
    np.save('/home/tza/STEAD/tza/x_EQdata_location_'+tag, X)
    np.save('/home/tza/STEAD/tza/y_EQlabelD_location_'+tag, y1)
    np.save('/home/tza/STEAD/tza/y_EQlabelP_location_'+tag, y2)
    np.save('/home/tza/STEAD/tza/y_EQlabelS_location_'+tag, y3)

    
#******************************Testing data preprocessing**************************************************


def normalize_test(data, mode = 'max'):  
    'Normalize waveforms in each batch'

    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data    


def data_generation_test(params, list_IDs):
    'readint the waveforms' 

    
    lengthT=len(list_IDs)
    X = np.zeros((lengthT, params['dim'], params['n_channels']))
    fl = h5py.File(params['file_name'], 'r')
    
    
    dataset = []
    with h5py.File(params['file_name'], 'r') as hdf5_file:
        # Get the list of event indices and sort them
        event_indices = list(hdf5_file.keys())
        sorted_event_indices = sorted(event_indices, key=get_event_number)

        for event_index in sorted_event_indices:
            index_number = get_event_number(event_index)

            if index_number in list_IDs:
                event_group = hdf5_file[event_index]
                event_waveforms = []
                for channel_index in event_group:
                    channel_data = np.array(event_group[channel_index])
                    event_waveforms.append(channel_data)
                dataset.append(event_waveforms)

    # Convert waveforms to a NumPy array
    dataset = np.array(dataset)
    waveforms = np.transpose(dataset, (0, 2, 1))
    
    
    # Generate data
    i=0
    for ID in enumerate(list_IDs):
        
        data = waveforms[i,:,:] 
        if params['norm_mode']:                    
            data = normalize_test(data, params['norm_mode']) #fdd 
        X[i, :, :] = data  
        i=i+1
    fl.close() 
     
    #X=X.astype('float32')
    
    lable="preprocessed_test"                                                             
    np.save('/home/tza/STEAD/tza/x_EQdata_location_'+lable, X)