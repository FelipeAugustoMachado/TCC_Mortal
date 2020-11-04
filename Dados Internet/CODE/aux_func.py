import numpy as np
from scipy.spatial.distance import pdist
from scipy.signal import butter, lfilter

### Filter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


### Correlation Dimension

def CorrelationDimension(data, num_points=100):
    dist_array = pdist(data)
    max_d = np.log(np.max(dist_array))*0.95
    min_d = np.log(np.min(dist_array))*1.05
    v_all = np.linspace(min_d, max_d, num_points)
    v_all = np.exp(v_all)
    n = data.shape[0]
    C2_values = np.zeros(v_all.size)
    for i in range(v_all.size):
        value = np.sum(dist_array<=v_all[i])/n 
        C2_values[i] = np.log(value)
    v = np.log(v_all)
    return C2_values, v

def derivate(data, v):
    v_d = v[1:-1]
    data_d = np.zeros(data.size-2)
    for i in range(1,data.size-1):
        data_d[i-1] = (data[i+1] - data[i-1])/(2*(v_d[1] - v_d[0]))
    return data_d, v_d


### Creating data

def createXY(data, label, w=10, t_class=400):
    X = np.zeros((data.shape[0]-w+1,data.shape[1]*w))
    Y = np.zeros(X.shape[0])
    for i in range(data.shape[0]-w+1):
        X[i,:] = data[i:i+w,:].flatten()
        Y[i] = label[i+w-1]
    return X, Y