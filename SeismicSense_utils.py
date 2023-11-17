 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 23:15:52 2020


@author: mostafamousavi
"""
from __future__ import print_function
import tensorflow as tk
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,add,UpSampling1D, Cropping1D, Reshape, Dense, Input,SpatialDropout1D, MaxPooling1D,TimeDistributed, Dropout, Activation, LSTM, Conv1D, Bidirectional, BatchNormalization, Concatenate
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
np.seterr(divide='ignore', invalid='ignore')
import h5py
from obspy.signal.trigger import trigger_onset
import pathlib

np.warnings.filterwarnings('ignore')
from tensorflow.keras import datasets, layers, models,Sequential








    
def f1(y_true, y_pred):
    
    """ 
    
    Calculate F1-score.
    
    Parameters
    ----------
    y_true : 1D array
        Ground truth labels. 
        
    y_pred : 1D array
        Predicted labels.     
        
    Returns
    -------  
    f1 : float
        Calculated F1-score. 
        
    """     
    
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    
############################################################################### 
###################################################################  Generator



def output_writter_test(args, 
                        dataset, 
                        evi, 
                        output_writer, 
                        csvfile, 
                        matches, 
                        pick_errors,
                        ):
    
    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.    
 
    dataset: hdf5 obj
        Dataset object of the trace.

    evi: str
        Trace name.    
              
    output_writer: obj
        For writing out the detection/picking results in the CSV file.
        
    csvfile: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        Contains the information for the detected and picked event.  
      
    pick_errors: dic
        Contains prediction errors for P and S picks.          
        
    Returns
    --------  
    X_test_results.csv  
    
        
    """        
    
    
    numberOFdetections = len(matches)
    
    if numberOFdetections != 0: 
        D_prob =  matches[list(matches)[0]][1]
        D_unc = matches[list(matches)[0]][2]

        P_arrival = matches[list(matches)[0]][3]
        P_prob = matches[list(matches)[0]][4] 
        P_unc = matches[list(matches)[0]][5] 
        P_error = pick_errors[list(matches)[0]][0]
        
        S_arrival = matches[list(matches)[0]][6] 
        S_prob = matches[list(matches)[0]][7] 
        S_unc = matches[list(matches)[0]][8]
        S_error = pick_errors[list(matches)[0]][1]  
        
    else: 
        D_prob = None
        D_unc = None 

        P_arrival = None
        P_prob = None
        P_unc = None
        P_error = None
        
        S_arrival = None
        S_prob = None 
        S_unc = None
        S_error = None
    
    if evi.split('_')[-1] == 'EV':                                     
        network_code = dataset.attrs['network_code']
        source_id = dataset.attrs['source_id']
        source_distance_km = dataset.attrs['source_distance_km']  
        snr_db = np.mean(dataset.attrs['snr_db'])
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = dataset.attrs['trace_start_time'] 
        source_magnitude = dataset.attrs['source_magnitude'] 
        p_arrival_sample = dataset.attrs['p_arrival_sample'] 
        p_status = dataset.attrs['p_status'] 
        p_weight = dataset.attrs['p_weight'] 
        s_arrival_sample = dataset.attrs['s_arrival_sample'] 
        s_status = dataset.attrs['s_status'] 
        s_weight = dataset.attrs['s_weight'] 
        receiver_type = dataset.attrs['receiver_type']  
                   
    elif evi.split('_')[-1] == 'NO':               
        network_code = dataset.attrs['network_code']
        source_id = None
        source_distance_km = None 
        snr_db = None
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = None
        source_magnitude = None
        p_arrival_sample = None
        p_status = None
        p_weight = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        receiver_type = dataset.attrs['receiver_type'] 
        
    if P_unc:
        P_unc = round(P_unc, 3)


    output_writer.writerow([network_code, 
                            source_id, 
                            source_distance_km, 
                            snr_db, 
                            trace_name, 
                            trace_category, 
                            trace_start_time, 
                            source_magnitude,
                            p_arrival_sample, 
                            p_status, 
                            p_weight, 
                            s_arrival_sample, 
                            s_status,
                            s_weight,
                            receiver_type, 
                            
                            numberOFdetections,
                            D_prob,
                            D_unc,    
                            
                            P_arrival, 
                            P_prob,
                            P_unc,                             
                            P_error,
                            
                            S_arrival, 
                            S_prob,
                            S_unc,
                            S_error,
                            
                            ]) 
    
    csvfile.flush() 

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind    
    

def picker(args, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, spt=None, sst=None):

    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
        Detection probabilities. 
        
    yh2 : 1D array
        P arrival probabilities.  
        
    yh3 : 1D array
        S arrival probabilities. 
        
    yh1_std : 1D array
        Detection standard deviations. 
        
    yh2_std : 1D array
        P arrival standard deviations.  
        
    yh3_std : 1D array
        S arrival standard deviations. 
        
    spt : {int, None}, default=None    
        P arrival time in sample.
        
    sst : {int, None}, default=None
        S arrival time in sample. 
        
   
    Returns
    --------    
    matches: dic
        Contains the information for the detected and picked event.            
        
    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
            
    pick_errors : dic                
        {detection statr-time:[ P_ground_truth - P_pick, S_ground_truth - S_pick]}
        
    yh3: 1D array             
        normalized S_probability                              
                
    """               
    
 #   yh3[yh3>0.04] = ((yh1+yh3)/2)[yh3>0.04] 
 #   yh2[yh2>0.10] = ((yh1+yh2)/2)[yh2>0.10] 
             
    detection = trigger_onset(yh1, args['detection_threshold'], args['detection_threshold'])
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)
          
    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    if len(pp_arr) > 0:
        P_uncertainty = None  
            
        for pick in range(len(pp_arr)): 
            pauto = pp_arr[pick]
                        
            if args['estimate_uncertainty'] and pauto:
                P_uncertainty = np.round(yh2_std[int(pauto)], 3)
                    
            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3) 
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})                 
                
    if len(ss_arr) > 0:
        S_uncertainty = None  
            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]
                   
            if args['estimate_uncertainty'] and sauto:
                S_uncertainty = np.round(yh3_std[int(sauto)], 3)
                    
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})             
            
    if len(detection) > 0:
        D_uncertainty = None  
        
        for ev in range(len(detection)):                                 
            if args['estimate_uncertainty']:               
                D_uncertainty = np.mean(yh1_std[detection[ev][0]:detection[ev][1]])
                D_uncertainty = np.round(D_uncertainty, 3)
                    
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
                    
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][1]]})            
    
    # matching the detection and picks
    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []
        
        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a,x] for x in l2[b:e]])
            
        best_pair = None
        for pr in ans: 
            ds = pr[1]-pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds           
        return best_pair


    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        S_error = None
        P_error = None        
        if int(ed-bg) >= 10:
                                    
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.update({Ss : S_val}) 
             
            if len(candidate_Ss) > 1:                
# =============================================================================
#                 Sr_st = 0
#                 buffer = {}
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     if S_valCan[0] > Sr_st:
#                         buffer = {SsCan : S_valCan}
#                         Sr_st = S_valCan[0]
#                 candidate_Ss = buffer
# =============================================================================              
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}


            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                        candidate_Ps.update({Ps : P_val}) 
                else:         
                    if Ps > bg-100 and Ps < ed:
                        candidate_Ps.update({Ps : P_val}) 
                    
            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan} 
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer
                    
            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}
                    
                    
# =============================================================================
#             Ses =[]; Pes=[]
#             if len(candidate_Ss) >= 1:
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     Ses.append(SsCan) 
#                                 
#             if len(candidate_Ps) >= 1:
#                 for PsCan, P_valCan in candidate_Ps.items():
#                     Pes.append(PsCan) 
#             
#             if len(Ses) >=1 and len(Pes) >= 1:
#                 PS = pair_PS(Pes, Ses, ed-bg)
#                 if PS:
#                     candidate_Ps = {PS[0] : candidate_Ps.get(PS[0])}
#                     candidate_Ss = {PS[1] : candidate_Ss.get(PS[1])}
# =============================================================================

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:                 
                matches.update({
                                bg:[ed, 
                                    EVENTS[ev][0], 
                                    EVENTS[ev][1], 
                                
                                    list(candidate_Ps)[0],  
                                    candidate_Ps[list(candidate_Ps)[0]][0], 
                                    candidate_Ps[list(candidate_Ps)[0]][1],  
                                                
                                    list(candidate_Ss)[0],  
                                    candidate_Ss[list(candidate_Ss)[0]][0], 
                                    candidate_Ss[list(candidate_Ss)[0]][1],  
                                                ] })
                
                if sst and sst > bg and sst < EVENTS[ev][2]:
                    if list(candidate_Ss)[0]:
                        S_error = sst -list(candidate_Ss)[0] 
                    else:
                        S_error = None
                                            
                if spt and spt > bg-100 and spt < EVENTS[ev][2]:
                    if list(candidate_Ps)[0]:  
                        P_error = spt - list(candidate_Ps)[0] 
                    else:
                        P_error = None
                                          
                pick_errors.update({bg:[P_error, S_error]})
      
    return matches, pick_errors, yh3


def _output_writter_test(args, 
                        dataset, 
                        evi, 
                        output_writer, 
                        csvfile, 
                        matches, 
                        pick_errors,
                        ):
    
    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.    
 
    dataset: hdf5 obj
        Dataset object of the trace.

    evi: str
        Trace name.    
              
    output_writer: obj
        For writing out the detection/picking results in the CSV file.
        
    csvfile: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        Contains the information for the detected and picked event.  
      
    pick_errors: dic
        Contains prediction errors for P and S picks.          
        
    Returns
    --------  
    X_test_results.csv  
    
        
    """        
    
    
    numberOFdetections = len(matches)
    
    if numberOFdetections != 0: 
        D_prob =  matches[list(matches)[0]][1]
        D_unc = matches[list(matches)[0]][2]

        P_arrival = matches[list(matches)[0]][3]
        P_prob = matches[list(matches)[0]][4] 
        P_unc = matches[list(matches)[0]][5] 
        P_error = pick_errors[list(matches)[0]][0]
        
        S_arrival = matches[list(matches)[0]][6] 
        S_prob = matches[list(matches)[0]][7] 
        S_unc = matches[list(matches)[0]][8]
        S_error = pick_errors[list(matches)[0]][1]  
        
    else: 
        D_prob = None
        D_unc = None 

        P_arrival = None
        P_prob = None
        P_unc = None
        P_error = None
        
        S_arrival = None
        S_prob = None 
        S_unc = None
        S_error = None
    
    if evi.split('_')[-1] == 'EV':                                     
        network_code = dataset.attrs['network_code']
        source_id = dataset.attrs['source_id']
        source_distance_km = dataset.attrs['source_distance_km']  
        snr_db = np.mean(dataset.attrs['snr_db'])
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = dataset.attrs['trace_start_time'] 
        source_magnitude = dataset.attrs['source_magnitude'] 
        p_arrival_sample = dataset.attrs['p_arrival_sample'] 
        p_status = dataset.attrs['p_status'] 
        p_weight = dataset.attrs['p_weight'] 
        s_arrival_sample = dataset.attrs['s_arrival_sample'] 
        s_status = dataset.attrs['s_status'] 
        s_weight = dataset.attrs['s_weight'] 
        receiver_type = dataset.attrs['receiver_type']  
                   
    elif evi.split('_')[-1] == 'NO':               
        network_code = dataset.attrs['network_code']
        source_id = None
        source_distance_km = None 
        snr_db = None
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = None
        source_magnitude = None
        p_arrival_sample = None
        p_status = None
        p_weight = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        receiver_type = dataset.attrs['receiver_type'] 
        
    if P_unc:
        P_unc = round(P_unc, 3)


    output_writer.writerow([network_code, 
                            source_id, 
                            source_distance_km, 
                            snr_db, 
                            trace_name, 
                            trace_category, 
                            trace_start_time, 
                            source_magnitude,
                            p_arrival_sample, 
                            p_status, 
                            p_weight, 
                            s_arrival_sample, 
                            s_status,
                            s_weight,
                            receiver_type, 
                            
                            numberOFdetections,
                            D_prob,
                            D_unc,    
                            
                            P_arrival, 
                            P_prob,
                            P_unc,                             
                            P_error,
                            
                            S_arrival, 
                            S_prob,
                            S_unc,
                            S_error,
                            
                            ]) 
    
    csvfile.flush()   
    
    
    


def _plotter(dataset, evi, args, save_figs, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, matches):
    

    """ 
    
    Generates plots.

    Parameters
    ----------
    dataset: obj
        The hdf5 obj containing a NumPy array of 3 component data and associated attributes.

    evi: str
        Trace name.  

    args: dic
        A dictionary containing all of the input parameters. 

    save_figs: str
        Path to the folder for saving the plots. 

    yh1: 1D array
        Detection probabilities. 

    yh2: 1D array
        P arrival probabilities.   
      
    yh3: 1D array
        S arrival probabilities.  

    yh1_std: 1D array
        Detection standard deviations. 

    yh2_std: 1D array
        P arrival standard deviations.   
      
    yh3_std: 1D array
        S arrival standard deviations. 

    matches: dic
        Contains the information for the detected and picked event.  
          
        
    """ 
    
    
    try:
        spt = int(dataset.attrs['p_arrival_sample']);
    except Exception:     
        spt = None
                    
    try:
        sst = int(dataset.attrs['s_arrival_sample']);
    except Exception:     
        sst = None

    predicted_P = []
    predicted_S = []
    if len(matches) >=1:
        for match, match_value in matches.items():
            if match_value[3]: 
                predicted_P.append(match_value[3])
            else:
                predicted_P.append(None)
                
            if match_value[6]:
                predicted_S.append(match_value[6])
            else:
                predicted_S.append(None)

    
    data = np.array(dataset)
    
    fig = plt.figure()
    ax = fig.add_subplot(411)

    plt.plot(data[:, 0], 'k')
    plt.subplots_adjust(hspace=0.50)
    plt.rcParams["figure.figsize"] = (8,5)
    legend_properties = {'weight':'bold'}  
    plt.title(str(evi), loc='left')
    #plt.tight_layout()
    ymin, ymax = ax.get_ylim() 
    pl = None
    sl = None       
    ppl = None
    ssl = None  
    
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='P arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='P arrival')
            
        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='S arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='S arrival')
        if pl or sl:    
            #plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
            plt.legend(loc='lower right', bbox_to_anchor=(0., 1.17, 1., .102), ncol=2, 
                       prop=legend_properties,  borderaxespad=0., fancybox=True, shadow=True)
                            
    ax = fig.add_subplot(412) 
    plt.plot(data[:, 1] , 'k')
    #plt.tight_layout() 
    
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='P arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='P arrival')
            
        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='S arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='S arrival')
        #if pl or sl:    
            #plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)    

    ax = fig.add_subplot(413) 
    plt.plot(data[:, 2], 'k')   
    #plt.tight_layout()                
    if len(predicted_P) > 0:
        ymin, ymax = ax.get_ylim()
        for pt in predicted_P:
            if pt:
                ppl = plt.vlines(int(pt), ymin, ymax, color='b', linewidth=2, label='P arrival')
    if len(predicted_S) > 0:  
        for st in predicted_S: 
            if st:
                ssl = plt.vlines(int(st), ymin, ymax, color='r', linewidth=2, label='S arrival')
                
    #if ppl or ssl:    
        #plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 

             
    ax = fig.add_subplot(414)
    x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
     
    if args['estimate_uncertainty']: 
        plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=1.5, label='Detection')
        lowerD = yh1-yh1_std
        upperD = yh1+yh1_std
        #plt.fill_between(x, lowerD, upperD, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')            
        
        plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=1.5, label='P probability')
        lowerP = yh2-yh2_std
        upperP = yh2+yh2_std
        #plt.fill_between(x, lowerP, upperP, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')  
                                     
        plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=1.5, label='S probability')
        lowerS = yh3-yh3_std
        upperS = yh3+yh3_std
        #plt.fill_between(x, lowerS, upperS, edgecolor='#CC4F1B', facecolor='#FF9848')
        
        plt.ylim((-0.1, 1.1))
        plt.tight_layout()
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                
    else:
        plt.plot(x, yh1, '--', color='g', alpha = 0.5, linewidth=1.5, label='Detection')
        plt.plot(x, yh2, '--', color='b', alpha = 0.5, linewidth=1.5, label='P_probability')
        plt.plot(x, yh3, '--', color='r', alpha = 0.5, linewidth=1.5, label='S_probability')
        plt.tight_layout()       
        plt.ylim((-0.1, 1.1))
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                        
    fig.savefig(os.path.join(save_figs, str(evi.split('/')[-1])+'.pdf')) 


    
############################################################# model######################
#Custom bidirectionalLSTM for micro-controller   
class BidirectionalLSTM(tk.keras.layers.Layer):
    def __init__(self, units, return_sequences=True, dropout=0.5, **kwargs):
        super(BidirectionalLSTM, self).__init__()
        self.units = units
        self.return_sequences = return_sequences
        self.dropout = dropout

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.lstm_forward = tk.keras.layers.LSTM(units=self.units, return_sequences=self.return_sequences, dropout=self.dropout)

    def call(self, inputs):
        # Reverse the input sequence along the time dimension
        inputs_reversed = inputs[:, ::-1, :]  # Reverse along the time dimension

        forward_output = self.lstm_forward(inputs)
        backward_output = self.lstm_forward(inputs_reversed)

        # Reverse the backward output along the time dimension again to match the original input
        backward_output_reversed = backward_output[:, ::-1, :]

        concatenated_output = tk.keras.layers.Concatenate(axis=-1)([forward_output, backward_output_reversed])
        return concatenated_output

    def get_config(self):
        config = super(BidirectionalLSTM, self).get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'dropout': self.dropout,
        })
        return config



#Custom Upsampling1DLayer for micro-controller
class Upsampling1DLayer(tk.keras.layers.Layer):
    def __init__(self, scale_factor,**kwargs):
        super(Upsampling1DLayer, self).__init__()
        self.scale_factor = scale_factor

    def get_config(self):
        config = super(Upsampling1DLayer, self).get_config()
        config.update({'scale_factor': self.scale_factor})
        return config

    def build(self, input_shape):
        super(Upsampling1DLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tk.shape(inputs)[0]
        time_steps = tk.shape(inputs)[1]
        num_channels = tk.shape(inputs)[2]

        reshaped_inputs = tk.expand_dims(inputs, axis=2)
        mask = tk.ones((1, 1, self.scale_factor, 1), dtype=reshaped_inputs.dtype)
        upsampled = reshaped_inputs * mask
        upsampled = tk.reshape(upsampled, (batch_size, time_steps * self.scale_factor, num_channels))

        return upsampled
    
    
#Custom CustomCropping1D for micro-controller
class CustomCropping1D(tk.keras.layers.Layer):
    def __init__(self, cropping_start, cropping_end, **kwargs):
        super(CustomCropping1D, self).__init__(**kwargs)
        self.cropping_start = cropping_start
        self.cropping_end = cropping_end

    def call(self, inputs):
        if self.cropping_end == 0:
            return inputs[:, self.cropping_start:, :]
        else:
            return inputs[:, self.cropping_start:-self.cropping_end, :]

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            batch_size = None
        else:
            batch_size = input_shape[0]
        seq_len = input_shape[1] - self.cropping_start - self.cropping_end
        return (batch_size, seq_len, input_shape[2])

    def get_config(self):
        config = super(CustomCropping1D, self).get_config()
        config.update({'cropping_start': self.cropping_start,
                       'cropping_end': self.cropping_end})
        return config


def generate_arrays_from_file(file_list, step):
  
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck   


def _block_CNN_1(filters, ker, drop_rate, activation, padding, inpC): 
    ' Returns CNN residual blocks '
    prev = inpC
    layer_1 = BatchNormalization()(prev) 
    act_1 = Activation(activation)(layer_1) 
    act_1 = SpatialDropout1D(drop_rate)(act_1)
    conv_1 = Conv1D(filters, ker, padding = padding)(act_1) 
    
    layer_2 = BatchNormalization()(conv_1) 
    act_2 = Activation(activation)(layer_2) 
    act_2 = SpatialDropout1D(drop_rate)(act_2)
    conv_2 = Conv1D(filters, ker, padding = padding)(act_2)
    
    res_out = add([prev, conv_2])
    
    return res_out

def _block_BiLSTM(filters, drop_rate, padding, inpR):
    'Returns LSTM residual block'    
    prev = inpR
    x_rnn = BidirectionalLSTM(filters, return_sequences=True, dropout=drop_rate)(prev)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out



def _encoder(filter_number, filter_size, depth, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the encoder that is a combination of residual blocks and maxpooling.'        
    e = inpC
    for dp in range(depth):
        e = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(e)             
        e = MaxPooling1D(2, padding = padding)(e)            
    return(e) 


def _decoder(filter_number, filter_size, depth, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the dencoder that is a combination of residual blocks and upsampling. '           
    d = inpC
    for dp in range(depth):        
        d = Upsampling1DLayer(2)(d) 
        if ((depth-dp) == 4) :
            d = CustomCropping1D(1, 1)(d) 
        if(depth-dp == 7 and dp!=0):
            d = CustomCropping1D(1, 1)(d)
        d = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(d)        
    return(d)

#****************helper functions*******************************

def lr_schedule(epoch):
    ' Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.'
    
    lr = 1e-3
    if epoch > 20:
        lr *= 0.5e-3
    elif epoch > 15:
        lr *= 1e-3
    elif epoch > 10:
        lr *= 1e-2
    elif epoch > 5:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr




class DataGenerator(tk.keras.utils.Sequence):
       
    def __init__(self,
                 data,
                 data_label1,
                 data_label2,
                 data_label3,
                 batch_size=32, 
                 shuffle=False ):
        
        'Initialization'
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data=data
        self.data_label1=data_label1
        self.data_label2=data_label2
        self.data_label3=data_label3
                
        self.list_IDs_len = len(self.data[:,0,0])
        
    def __len__(self):
        #print('*************_len_')
        'Denotes the number of batches per epoch'
        return int(np.floor(self.list_IDs_len / self.batch_size))

    def __getitem__(self, index):

        #print('*************on_batch_end')
        X=self.data[index*self.batch_size:(index+1)*self.batch_size,:,:]
        y1=self.data_label1[index*self.batch_size:(index+1)*self.batch_size,:]
        y2=self.data_label2[index*self.batch_size:(index+1)*self.batch_size,:]
        y3=self.data_label3[index*self.batch_size:(index+1)*self.batch_size,:]

        return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})
    
    
class DataGeneratorPrediction(tk.keras.utils.Sequence):
  
    
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim, 
                 batch_size=32, 
                 n_channels=3, 
                 norm_mode = 'max'):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def normalize(self, data, mode = 'max'): 
        'Normalize waveforms in a batch'
         
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
 
    def __data_generation(self, list_IDs_temp):
        'read the waveforms'         
        X = np.zeros((self.batch_size, self.dim, self.n_channels))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            dataset = fl.get('data/'+str(ID))
            data = np.array(dataset)                
         
            if self.norm_mode:                    
                data = self.normalize(data, self.norm_mode)  
                            
            X[i, :, :] = data                                       

        fl.close() 
                           
        return X
        

            
class SeismicSense_original():
    
    """ 
    
    Creates the model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.

    activationf: str
        Activation funciton type.

    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.

    cnn_blocks: int
        The number of residual CNN blocks.

    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.

    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 

    kernel_regularizer: str
        l1 norm regularizer.

    bias_regularizer: str
        l1 norm regularizer.

    multi_gpu: bool
        If use multiple GPUs for the training. 

    gpu_number: int
        The number of GPUs for the muli-GPU training. 
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 8, 16, 32, 32, 32, 32],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4, 
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x) 
        
        
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(self.nb_filters[6], 3, self.drop_rate, self.activationf, self.padding, x)
            if cb > 2:
                x = _block_CNN_1(self.nb_filters[6], 2, self.drop_rate, self.activationf, self.padding, x)

        for bb in range(self.BiLSTM_blocks):
            x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)
            
        x= LSTM(64, return_sequences=True, dropout=self.drop_rate)(x)
    
        
        encoded = BatchNormalization(name='encoder_block')(x)    
        
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)
        

        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            encoded)
        
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            encoded) 
        
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        model = Model(inputs=inp, outputs=[d, P, S])
        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(lr=lr_schedule(0)), metrics=[f1])

        return model

 