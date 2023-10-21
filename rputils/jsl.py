"""
Functions to read text files of waveforms generated 
by JSeisLab software
"""

import numpy as np
from scipy.signal import detrend



def load_waveform(filepath, apply_detrend=True, return_header=False):
    """
    Read ASCII file created by JSeisLab and create a numpy
    array of dimension [n samples, 2]. Dimension [:, 0] is time,
    dimension [:, 1] is amplitude
    
    
    filepath (str): directory location of ASCII file
    apply_detrend (bool): detrend signal amplitude
    return_header (bool): optionally return dictionary of
        ASCII file headers
    """
    
    with open(filepath) as f:
        
        ## Read file, skip headers to grab data
        lines = f.readlines()
        wavelines = lines[3:]
        
        ## Gather file headers
        header1 = lines[0]
        header2 = lines[1]
        header1 = header1.split(", ")
        header2 = header2.split(", ")
        headers = {}
        for i in range(len(header1)):
            headers[header1[i]] = header2[i]

        ## Initialize blank array, read each sample from
        ## ASCII file and format into the array
        waveform = np.zeros((2, len(wavelines)))
        for i in range(len(wavelines)):
            waveline = wavelines[i]
            time, amp = waveline.split(", ")
            time = float(time.replace("+", ""))
            amp = float(amp.replace("+", ""))  

            waveform[0, i] = time
            waveform[1, i] = amp

    ## Trigger offset is stored in odd location, grab it
    ## and correct the wavetime
    trigger_offset = float(headers["T outside"]) *1e9
    waveform[0,:] = waveform[0,:] + trigger_offset
    
    ## Detrend amplitude data to remove bias
    if apply_detrend:
        ## Detrended amplitude
        detrend_amp = detrend(waveform[1,:])
        waveform[1,:] = detrend_amp
        
    if return_header:
        return waveform, headers
    else:
        return waveform



