from tkinter.constants import N, NO
# from appscript import k
import numpy as np
from scipy import signal, optimize
from common import config as cfg
from numba import jit
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from collections import Iterable

def calc_phase_diff(a,b):
    diff = np.angle(a) - np.angle(b)
    diff -= (diff>=np.pi)*np.pi*2
    diff += (diff<=-np.pi)*np.pi*2
    return diff

def reshape_data(data,tx=1,rx=4):
    if(tx==3):
        return data
    elif(tx==1):
        result = np.zeros((int(data.shape[0]*data.shape[1]/rx),rx,data.shape[2]),dtype=complex)
        # print(result.shape)
        for i in range(data.shape[0]):
            for j in range(rx):
                for k in range(int(data.shape[1]/rx)):
                    result[int(i*data.shape[1]/rx)+k,j,:] = data[i,k*rx+j,:]
        return result                
    else:
        print('error!tx=',tx)
    

def calc_pd(x,m=20):
    min_sample = x.min()
    max_sample = x.max()
    x_index = np.zeros_like(x)
    max_sample = min_sample + 1.01*(max_sample-min_sample)
    unit = (max_sample - min_sample)/m
    # for i in range(x.shape[0]):
    #     if(i%10==0):
    #         print(i,x.shape[0])
    #     for j in range(x.shape[1]):
    #         for k in range(m):
    #             if(x[i,j]>=(min_sample+k*unit) and x[i,j]<(min_sample+(k+1)*unit)):
    #                 x_index[i,j] = k
    for i in range(m):
        cur_index = x>=min_sample+i*unit
        x_index[cur_index] = i
    return x_index

def calc_ent(*cin):
    x = cin[0]
    # calc joint shannon entropy
    if(len(cin)==2):
        y = cin[1]
        x_value_list = set([x[i] for i in range(x.shape[0])])
        y_value_list = set([y[i] for i in range(y.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            for y_value in y_value_list:
                p = float(x[x == x_value].shape[0]) * float(y[y == y_value].shape[0]) / x.shape[0] / y.shape[0]
                ent -= p * np.log2(p)
        return ent

    # calc shannon entropy of x
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        ent -= p * np.log2(p)
    return ent

def calc_mi(x,y):
    return calc_ent(x)+calc_ent(y)-calc_ent(x,y)      
    
def fft(data: ndarray, axis=-1, window=None, shift=False, half=False):
    data = data.swapaxes(axis, -1)
    if window:
        data = data * window(data.shape[-1])
    result: ndarray = np.fft.fft(data)
    if half:
        result = result[..., :(result.shape[-1] + 1) // 2]
    elif shift:
        result = np.fft.fftshift(result, axes=(-1, ))
    return result.swapaxes(axis, -1)


def beamforming(data: ndarray, num=4, from_=-np.pi, to=np.pi):
    offset = np.linspace(from_, to, num=num, endpoint=False)
    d = np.exp(-1j * np.arange(0, data.shape[-2]) * offset[:, np.newaxis])
    return np.matmul(d, data), offset


def range_fft(data: ndarray, axis=2, window=np.hanning, bins=None, ranges=None):
    data = fft(data, axis=axis, window=window).swapaxes(0, axis)
    range_bins = cfg.dca.range_bins
    return data, range_bins


def multiview_range_fft(data: ndarray, num_views=8, interval=8, axis=2, window=np.hanning, bins=None, ranges=None):
    window_len = cfg.dca.no_samples-(num_views-1)*interval
    sliding = sliding_window_view(data, window_len, axis)

    if axis >= 0:
        axis += 1
    sliding = sliding.swapaxes(0, axis).swapaxes(1, axis-1).swapaxes(0, 1)
    sliding = sliding[::interval]
    data = fft(sliding, axis=1, window=window)
    range_bins = np.arange(0, cfg.dca.no_samples) * 299792458 / \
        (2.0 * cfg.dca.bandwidth * window_len/cfg.dca.no_samples)
    return data, range_bins


def range2bin(r, bins: ndarray = cfg.dca.range_bins):
    diff = r-bins[..., np.newaxis]
    diff[diff < 0] = 9999999
    return np.argmin(diff, axis=0)


def doppler_fft(data: ndarray, axis=-1, window=np.hanning):
    return fft(data, axis=axis, window=window, shift=True)


def angle_fft(data: ndarray, axis=1, window=None):
    return fft(data, axis=axis, window=window, shift=True)


def circle_fit(data: ndarray):
    def target(x, data):
        z = x[0] + 1j * x[1]
        r = x[2]
        return np.sum(np.abs(np.abs(data - z) - r))

    res = optimize.least_squares(target, np.array([0, 0, 0]), args=(data, ))
    x = res.x
    return x[0] + 1j * x[1], x[2]


def stft(data: ndarray, nperseg, interval=5, zoom=None, fs=cfg.dca.fs, axis=-1):
    f, t, zxx = signal.stft(data.swapaxes(axis, -1),
                            fs=fs,
                            nperseg=nperseg,
                            noverlap=nperseg - interval,
                            detrend='linear',
                            return_onesided=False,
                            boundary='odd')
    f: ndarray = np.fft.fftshift(f)
    zxx: ndarray = np.fft.fftshift(zxx, axes=axis - 1)

    if zoom:
        center = np.squeeze(np.argwhere(f == 0))
        f = f[center - zoom + 1:center + zoom + 1]
        zxx = zxx[..., center - zoom + 1:center + zoom, :]
    return f, t, zxx


def stft_heatmap(data: ndarray,
                 nperseg,
                 interval=5,
                 normalize=True,
                 zoom=None,
                 fs=cfg.dca.fs,
                 axis=-1):
    f, t, zxx = stft(data, nperseg, interval, zoom, fs, axis)
    zxx = np.abs(zxx)
    if normalize:
        zxx /= np.mean(zxx, axis=(-1, -2))[..., np.newaxis, np.newaxis]
    return f, t, zxx


def fir_bandpass_design(fl, fh, numtaps=1001, fs=cfg.dca.fs):
    f = np.arange(fs // 2+1)
    if len(f) % 2:
        f = f[:-1]
    flag = np.zeros_like(f)

    f[int(fl)] = fl
    while True:
        try:
            f[int(fh)] = fh
            break
        except IndexError:
            fh -= 1

    flag[int(fl):int(fh)+1] = 1

    return signal.firls(numtaps, f, flag, fs=fs)


def filt(data: ndarray, filter: ndarray, axis=-1, mode='valid', reverse=False):
    data = data.swapaxes(axis, -1)
    if reverse:
        data = data[..., ::-1]
    new_shape = list(filter.shape)
    new_shape = [1] * (len(data.shape) - len(filter.shape)) + new_shape

    result = signal.convolve(data, filter.reshape(new_shape), mode=mode)
    if reverse:
        result = result[..., ::-1]
    return result.swapaxes(axis, -1)


def fir_bandpass_filt(data: ndarray,
                      fl,
                      fh,
                      axis=-1,
                      numtaps=1001,
                      fs=cfg.dca.fs,
                      mode='valid',
                      reverse=False):
    filter = fir_bandpass_design(fl, fh, fs=fs, numtaps=numtaps)
    return filt(data, filter, axis, mode=mode, reverse=reverse)


def second_derivative_filt(data: ndarray, axis=-1, mode='valid'):
    filter = np.array([-1, -2, 1, 4, 1, -2, -1])
    return filt(data, filter, axis, mode=mode)


@jit(nopython=True)
def hampel_filt(data: ndarray, window_size=20, n_sigmas=1):
    data = np.ascontiguousarray(data)
    shape = data.shape
    n = shape[-1]
    data = data.reshape((-1, n))
    new_data = data.copy()
    k = 1.4826

    for j in range(new_data.shape[0]):
        for i in range((window_size), (n - window_size)):
            x0 = np.median(data[j, (i - window_size):(i + window_size)])
            S0 = k * np.median(
                np.abs(data[j, (i - window_size):(i + window_size)] - x0))
            if (np.abs(data[j, i] - x0) > n_sigmas * S0):
                new_data[j, i] = x0

    return new_data.reshape(shape)


def ca_cfar(data: ndarray, guard_len, noise_len, weight=1, axis=-1, mode='valid'):
    data = data.swapaxes(axis, -1)
    kernel = np.ones(1 + (2 * guard_len) +
                     (2 * noise_len), dtype=data.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0
    kernel *= weight
    threshold = filt(data, kernel, mode=mode)
    return threshold.swapaxes(axis, -1)


def outlier_detect(x: ndarray, p=3):
    upbound = np.percentile(
        x, 75) + p * (np.percentile(x, 75) - np.percentile(x, 25))
    downbound = np.percentile(
        x, 25) - p * (np.percentile(x, 75) - np.percentile(x, 25))
    return (x > upbound) | (x < downbound)


def complex_int16(x: ndarray):
    return np.ascontiguousarray(x.astype(np.complex64)).view(np.float32).astype(np.int16)


def int16_complex(x: ndarray):
    return np.ascontiguousarray(x.astype(np.float32)).view(np.complex64)


