"""
Assorted useful filters
"""

import numpy as np
from scipy.signal import butter, lfilter
from scipy.ndimage import generic_filter



def bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def lowpass(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='lowpass', analog=False)
    y = lfilter(b, a, data)
    return y

def highpass(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='highpass', analog=False)
    y = lfilter(b, a, data)
    return y

    
    
def median(arr, size=5, iterations=1):
    """
    From Bruges library  https://github.com/agilescientific/bruges
    A nonlinear n-D edge-preserving smoothing filter.
    
    Assumes imports:
        import numpy as np
        from scipy.ndimage import generic_filter
    
    Args:
        arr (ndarray): an n-dimensional array, such as a seismic horizon.
        size (int): the kernel size, e.g. 5 for 5x5. Should be odd,
            rounded up if not.
    Returns:
        ndarray: the resulting smoothed array.
    """
    arr = np.array(arr, dtype=float)

    if not size // 2:
        size += 1
    filtered = generic_filter(arr, np.median, size=size)

    return filtered