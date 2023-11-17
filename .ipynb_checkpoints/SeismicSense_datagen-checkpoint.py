import tensorflow as tf
import numpy as np
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

def data_generation(params, list_IDs,tag):
    'read the waveforms' 
    lengthT=len(list_IDs)
    indexes = np.arange(lengthT)
    if params['shuffle'] == True:
        np.random.shuffle(indexes)
    if params['augmentation']== True:
        indexes = np.append(indexes, indexes) 
        lengthT=2*lengthT
        
    list_IDs_temp = [list_IDs[k] for k in indexes]
  
    X = np.zeros((lengthT, params['dim'], params['n_channels']))
    y1 = np.zeros((lengthT, params['dim'], 1))
    y2 = np.zeros((lengthT, params['dim'], 1))
    y3 = np.zeros((lengthT, params['dim'], 1))
    fl = h5py.File(params['file_name'], 'r')

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        additions = None
        dataset = fl.get('data/'+str(ID))

        if ID.split('_')[-1] == 'EV':
            data = np.array(dataset)                    
            spt = int(dataset.attrs['p_arrival_sample']);
            sst = int(dataset.attrs['s_arrival_sample']);
            coda_end = int(dataset.attrs['coda_end_sample']);
            snr = dataset.attrs['snr_db'];

        elif ID.split('_')[-1] == 'NO':
            data = np.array(dataset)

        ## augmentation 
        if params['augmentation'] == True:                 
            if i <= lengthT//2:   
                if params['shift_event_r'] and dataset.attrs['trace_category'] == 'earthquake_local':
                    data, spt, sst, coda_end = shift_event(data, spt, sst, coda_end, snr, params['shift_event_r']/2);                                       
                if params['norm_mode']:                    
                    data = normalize(data, params['norm_mode'])  
            else:                  
                if dataset.attrs['trace_category'] == 'earthquake_local':                   
                    if params['shift_event_r']:
                        data, spt, sst, coda_end = shift_event(data, spt, sst, coda_end, snr, params['shift_event_r']); 

                    if params['add_event_r']:
                        data, additions = add_event(data, spt, sst, coda_end, snr, params['add_event_r']); 

                    if params['add_noise_r']:
                        data = add_noise(data, snr, params['add_noise_r']);

                    if params['drop_channel_r']:    
                        data = drop_channel(data, snr, params['drop_channel_r']);
                        data = adjust_amplitude_for_multichannels(data)  

                    if params['scale_amplitude_r']:
                        data = scale_amplitude(data, params['scale_amplitude_r']); 

                    if params['pre_emphasis']:  
                        data = pre_emphasis(params, data) 

                    if params['norm_mode']:    
                        data = normalize(data, params['norm_mode'])                            

                elif dataset.attrs['trace_category'] == 'noise':
                    if params['drop_channel_r']:    
                        data = drop_channel_noise(data, params['drop_channel_r']);

                    if params['add_gap_r']:    
                        data = add_gaps(data, params['add_gap_r'])

                    if params['norm_mode']: 
                        data = normalize(data, params['norm_mode']) 

        elif params['augmentation'] == False:  
            if params['shift_event_r'] and dataset.attrs['trace_category'] == 'earthquake_local':
                data, spt, sst, coda_end = shift_event(data, spt, sst, coda_end, snr, params['shift_event_r']/2);                     
            if params['norm_mode']:                    
                data = normalize(data, params['norm_mode'])                          

        X[i, :, :] = data                                       

        ## labeling 
        if dataset.attrs['trace_category'] == 'earthquake_local': 
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

                if additions: 
                    add_sd = None
                    add_spt = additions[0];
                    add_sst = additions[1];
                    if add_spt and add_sst: 
                        add_sd = add_sst - add_spt  

                    if add_sd and add_sst+int(0.4*add_sd) <= params['dim']: 
                        y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1        
                    else:
                        y1[i, add_spt:params['dim'], 0] = 1

                    if add_spt and (add_spt-20 >= 0) and (add_spt+20 < params['dim']):
                        y2[i, add_spt-20:add_spt+20, 0] = np.exp(-(np.arange(add_spt-20,add_spt+20)-add_spt)**2/(2*(10)**2))[:params['dim']-(add_spt-20)]
                    elif add_spt and (add_spt+20 < params['dim']):
                        y2[i, 0:add_spt+20, 0] = np.exp(-(np.arange(0,add_spt+20)-add_spt)**2/(2*(10)**2))[:params['dim']-(add_spt-20)]

                    if add_sst and (add_sst-20 >= 0) and (add_sst+20 < params['dim']):
                        y3[i, add_sst-20:add_sst+20, 0] = np.exp(-(np.arange(add_sst-20,add_sst+20)-add_sst)**2/(2*(10)**2))[:params['dim']-(add_sst-20)]
                    elif add_sst and (add_sst+20 < params['dim']):
                        y3[i, 0:add_sst+20, 0] = np.exp(-(np.arange(0,add_sst+20)-add_sst)**2/(2*(10)**2))[:params['dim']-(add_sst-20)]


            elif params['label_type']  == 'triangle':                      
                sd = None    
                if spt and sst: 
                    sd = sst - spt  

                if sd and sst:
                    if sst+int(0.4*sd) <= params['dim']: 
                        y1[i, spt:int(sst+(0.4*sd)), 0] = 1        
                    else:
                        y1[i, spt:params['dim'], 0] = 1                     

                if spt and (spt-20 >= 0) and (spt+21 < params['dim']):
                    y2[i, spt-20:spt+21, 0] = label()
                elif spt and (spt+21 < params['dim']):
                    y2[i, 0:spt+spt+1, 0] = label(a=0, b=spt, c=2*spt)
                elif spt and (spt-20 >= 0):
                    pdif = params['dim'] - spt
                    y2[i, spt-pdif-1:params['dim'], 0] = label(a=spt-pdif, b=spt, c=2*pdif)

                if sst and (sst-20 >= 0) and (sst+21 < params['dim']):
                    y3[i, sst-20:sst+21, 0] = label()
                elif sst and (sst+21 < params['dim']):
                    y3[i, 0:sst+sst+1, 0] = label(a=0, b=sst, c=2*sst)
                elif sst and (sst-20 >= 0):
                    sdif = params['dim'] - sst
                    y3[i, sst-sdif-1:params['dim'], 0] = label(a=sst-sdif, b=sst, c=2*sdif)             

                if additions: 
                    add_spt = additions[0];
                    add_sst = additions[1];
                    add_sd = None
                    if add_spt and add_sst: 
                        add_sd = add_sst - add_spt                     

                    if add_sd and add_sst+int(0.4*add_sd) <= params['dim']: 
                        y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1        
                    else:
                        y1[i, add_spt:params['dim'], 0] = 1                     

                    if add_spt and (add_spt-20 >= 0) and (add_spt+21 < params['dim']):
                        y2[i, add_spt-20:add_spt+21, 0] = label()
                    elif add_spt and (add_spt+21 < params['dim']):
                        y2[i, 0:add_spt+add_spt+1, 0] = label(a=0, b=add_spt, c=2*add_spt)
                    elif add_spt and (add_spt-20 >= 0):
                        pdif = params['dim'] - add_spt
                        y2[i, add_spt-pdif-1:params['dim'], 0] = label(a=add_spt-pdif, b=add_spt, c=2*pdif)

                    if add_sst and (add_sst-20 >= 0) and (add_sst+21 < params['dim']):
                        y3[i, add_sst-20:add_sst+21, 0] = label()
                    elif add_sst and (add_sst+21 < params['dim']):
                        y3[i, 0:add_sst+add_sst+1, 0] = label(a=0, b=add_sst, c=2*add_sst)
                    elif add_sst and (add_sst-20 >= 0):
                        sdif = params['dim'] - add_sst
                        y3[i, add_sst-sdif-1:params['dim'], 0] = label(a=add_sst-sdif, b=add_sst, c=2*sdif) 


            elif params['label_type']  == 'box':
                sd = None                             
                if sst and spt:
                    sd = sst - spt      

                if sd and sst+int(0.4*sd) <= params['dim']: 
                    y1[i, spt:int(sst+(0.4*sd)), 0] = 1        
                else:
                    y1[i, spt:params['dim'], 0] = 1         
                if spt: 
                    y2[i, spt-20:spt+20, 0] = 1
                if sst:
                    y3[i, sst-20:sst+20, 0] = 1                       

                if additions:
                    add_sd = None
                    add_spt = additions[0];
                    add_sst = additions[1];
                    if add_spt and add_sst:
                        add_sd = add_sst - add_spt  

                    if add_sd and add_sst+int(0.4*add_sd) <= params['dim']: 
                        y1[i, add_spt:int(add_sst+(0.4*add_sd)), 0] = 1        
                    else:
                        y1[i, add_spt:params['dim'], 0] = 1                     
                    if add_spt:
                        y2[i, add_spt-20:add_spt+20, 0] = 1
                    if add_sst:
                        y3[i, add_sst-20:add_sst+20, 0] = 1                 

    fl.close()
                                                                    
    X=X.astype('float32')
    y1=y1.astype('float32')
    y2=y2.astype('float32')
    y3=y3.astype('float32')
                                 
                                                                    
    np.save('/home/tza/STEAD/tza/x_EQdata_'+tag, X)
    np.save('/home/tza/STEAD/tza/y_EQlabelD_'+tag, y1)
    np.save('/home/tza/STEAD/tza/y_EQlabelP_'+tag, y2)
    np.save('/home/tza/STEAD/tza/y_EQlabelS_'+tag, y3)

    
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

    # Generate data
    for i, ID in enumerate(list_IDs):
        if ID.split('_')[-1] == 'EV':
            dataset = fl.get('data/'+str(ID))
            data = np.array(dataset)              

        elif ID.split('_')[-1] == 'NO':
            dataset = fl.get('data/'+str(ID))
            data = np.array(dataset)

        if params['norm_mode']:                    
            data = normalize_test(data, params['norm_mode'])  

        X[i, :, :] = data                                       

    fl.close() 
     
    #X=X.astype('float32')
    
    lable="preprocessed_test"                                                             
    np.save('/home/tza/STEAD/tza/x_EQdata_'+lable, X)