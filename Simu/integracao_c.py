#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

from controle import *
from threading import Thread
from scipy.signal import lfilter

### Func aux

def importFiles(version):
    if version == 'pca':
        # PCA object
        with open('files/pca_obj.pkl', 'rb') as f:
            pca = pickle.load(f)

        # PCA model
        with open('files/model_pca.pkl', 'rb') as f:
            model = pickle.load(f)

        return pca, model

    elif version == 'ica':
        # ICA object
        with open('files/ica_obj.pkl', 'rb') as f:
            ica = pickle.load(f)

        # ICA model
        with open('files/model_ica.pkl', 'rb') as f:
            model = pickle.load(f)

        return ica, model

    elif version == 'cca':
        # SOM centers 
        centers = pd.read_parquet('files/map.parquet').values

        # CCA centers
        with open('files/cca_centers.pkl', 'rb') as f:
            cca_centers = pickle.load(f)

        # CCA model
        with open('files/model_cca.pkl', 'rb') as f:
            model = pickle.load(f)

        return centers, cca_centers, model

    else:
        return None



def featureCreate(data, files):
    # ICA and PCA
    if len(files) == 2:
        transform, model = files
        data_trans = transform.transform(data)
        return model.predict_proba(data_trans.flatten().reshape(1,-1))[:,1]

    # To do
    else:
        centers, cca, model = files
        idx = np.apply_along_axis(lambda r: np.linalg.norm(r - centers, axis=1).argmin(), 1, data)
        data_trans = cca[idx].flatten().reshape(1,-1)
        return model.predict_proba(data_trans)[:,1]



### Variables

# Global
t0 = time.time()
output_model = 0
y = [0]
time_list = [0]

# Control
control_model = HLC()


# Pre-process
with open('files/filter_b_a.pkl', 'rb') as f:
    b,a = pickle.load(f)

version = 'pca'
threshold = 0.32
files = importFiles(version)
data = pd.read_parquet('files/data3pp_10.parquet')
X = data.drop('Classe', axis=1).values
#y = data['Classe'].values

### Functions Parallel


def preprocess():
    global output_model

    model_output = np.zeros((len(data) - 10, 2))
    runs = len(data) - 10
    for i in range(runs):
        sample = X[i:i+10]
        s1 = time.time()
        sample_f = lfilter(b, a, sample)
        result = featureCreate(sample_f, files)[0]
        s2 = time.time()
        output_model = (result > threshold).astype(int)
        while (s2-s1) < 0.0038:
            s2 = time.time()
        print(round(result,2), end='\r')
    output_model = -1
        

# def model():
#     global output_model
    
#     for i in range(len(output_list)):
#         output_model = output_list[i]
#         time.sleep(dt)
#     print('Stop')


def control():
    global output_model
    global y
    global time_list
    
    while output_model != -1:
        if output_model == 1 and control_model.state == -1:
            time_add = np.arange(time_list[-1], time.time() - t0, 0.05).tolist()
            y_add = (np.ones(len(time_add))*y[-1]).tolist()
            
            time_list += time_add
            y += y_add
            
            s1 = time.time()
            control_model.Do_HLC(1)
            s2 = time.time()
            while s2-s1<1.55:
                s2 = time.time()
            t1 = time.time()
            
            y += control_model.llc.y_evolution
            time_list += (np.array(control_model.llc.t_evolution) + (s1 - t0)).tolist()
            
        if control_model.state == 1:
            t2 = time.time()
            if (t2 - t1) > 1.5:
                
                time_add = np.arange(time_list[-1], time.time() - t0, 0.05).tolist()
                y_add = (np.ones(len(time_add))*y[-1]).tolist()

                time_list += time_add
                y += y_add
                
                s1 = time.time()
                control_model.Do_HLC(-1)
                s2 = time.time()
                while s2-s1<1.55:
                    s2 = time.time()
                y += (np.array(control_model.llc.y_evolution) + y[-1]).tolist()
                time_list += (np.array(control_model.llc.t_evolution) + (s1 - t0)).tolist()
        
        time.sleep(0.05)
    
def plot():
    global y
    global time_list
    global output_model
    
    time.sleep(2)
    
    while output_model != -1:
        
        time2plot = np.array(time_list)# + [time.time() - t0])
        y2plot = np.array(y)# + [y[-1]])
        
        time_init = (time.time() - t0) - 9.6 #s
        time_final = (time.time() - t0) - 1.6 #s
        
        cond = (time2plot > time_init) & (time2plot < time_final)
        
        if len(time2plot[cond]) == 0:
            time2plot = np.arange(time_init, time_final, 0.05).tolist()
            y2plot = (np.ones(len(time2plot))*y[-1]).tolist()

        else:
            time2plot = time2plot[cond].tolist() + [time_final]
            y2plot = y2plot[cond].tolist() + [y2plot[cond][-1]]
            
        plt.plot(time2plot, y2plot)
        plt.xlabel('Time (ms)')
        plt.ylabel('Braco')
        plt.ylim([-0.1,1.5])
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        time.sleep(0.05)
    
    

def main():
    p1 = Thread(target = preprocess)
    p2 = Thread(target = control)
    p3 = Thread(target = plot)
    p1.start()
    p2.start()
    p3.start()


# In[3]:


main()

