# flake8: noqa
# type: ignore
import numpy as np
from pathlib import Path
import os
import psutil


from scipy.sparse import data

user = 'zs'
feature_path = Path(r'.\feature')
time_format = r'%y-%m-%d %H-%M-%S'

def showstorage(index=1):
    print(index,u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) ) 

class dca:
    direct = False
    filename = 'dca.npy'
    ip = '192.168.33.30'
    port = 4098
    freq_slope = 29.982
    freq_start = 60
    no_antennas = 12
    no_chirps = 1
    no_samples = 256
    sample_rate = 10000
    chirp_time = 833.335
    # calculated parameters
    fs = 1200
    wavelength = 0.3 / freq_start

    bandwidth = freq_slope / sample_rate * no_samples * 1e9
    range_resolution = 299792458 / (2.0 * bandwidth)

    range_bins = np.arange(0, no_samples) * range_resolution

    angle_bins = np.rad2deg(
        np.arcsin(2 * np.fft.fftshift(np.fft.fftfreq(no_antennas))))

class mydca:
    no_tx = 3
    no_rx = 4
    no_chirp = 192
    preriodicity = 0.5
    freq_start = 60
    wavelength = 0.3 / freq_start
    bandwidth = 4
    range_bin = 3 / 20 / bandwidth
    doppler_bin = wavelength / 2 / (no_chirp*preriodicity) * 1000
    rx_index = [[7,6,11,10],[4,5,8,9],[7,4,3,0],[6,5,2,1]]#x,x,y,y
    doppler_bin_list = np.round(np.linspace(-no_chirp/2,no_chirp/2,no_chirp,endpoint=False)*doppler_bin,2)
    time_length = 0
    time_bin_list = np.array([])

if __name__=='__main__':
    print(mydca.doppler_bin_list,mydca.doppler_bin)