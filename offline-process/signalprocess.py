from cmath import pi
import common.config as cfg
from common import fmcw
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
import seaborn as sns
from sklearn import metrics
import gc
import time
import psutil
import os
import showresult as sr
import cv2
# the returned array is like:(channel,x,y,time)

def show_rp(filename,remove_static=1,slides=cfg.mydca.no_chirp,toprint=1):
    if(toprint==1):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating range profile')
    # calc range profile
    raw_data = np.load(filename)
    data = fmcw.int16_complex(raw_data)
    range_fft_data,_ = fmcw.range_fft(data)
    range_fft_data = range_fft_data[:,:,:int(range_fft_data.shape[2]/slides)*slides]
    if(remove_static==1):
        for i in range(int(range_fft_data.shape[2]/slides)):
            for j in range(range_fft_data.shape[0]):
                for k in range(range_fft_data.shape[1]):
                    range_fft_data[j,k,i*slides:(i+1)*slides] = range_fft_data[j,k,i*slides:(i+1)*slides] - range_fft_data[j,k,i*slides:(i+1)*slides].mean()
    
    return range_fft_data.transpose((1,0,2))[:,:,:,np.newaxis]

def show_rdp(filename,slides=cfg.mydca.no_chirp,threshold=0,remove_static=1,toprint=1):
    if(toprint==1):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating range doppler profile')
    # calc range doppler profile
    raw_data = np.load(filename)
    # print(raw_data.shape)
    data = fmcw.int16_complex(raw_data)
    range_fft_data,_ = fmcw.range_fft(data)
    range_fft_data = range_fft_data[:,:,:int(range_fft_data.shape[2]/slides)*slides]
    
    num_chirp = cfg.mydca.no_chirp
    num_frame = int((range_fft_data.shape[-1]-num_chirp)/slides) + 1
    doppler_fft_data = np.zeros((num_frame,range_fft_data.shape[0],range_fft_data.shape[1],num_chirp),dtype=complex)
    for i in range(num_frame):
        if(remove_static==1):
            doppler_fft_data[i] = fmcw.doppler_fft(range_fft_data[:,:,i*slides:i*slides+num_chirp]-range_fft_data[:,:,i*slides:i*slides+num_chirp].mean(2).reshape((range_fft_data.shape[0],range_fft_data.shape[1],1)))
        else:
            doppler_fft_data[i] = fmcw.doppler_fft(range_fft_data[:,:,i*slides:i*slides+num_chirp])

    # filter < threshold
    if(threshold!=0):
        max_amp = abs(doppler_fft_data).max()
        min_amp = abs(doppler_fft_data).min()
        threshold_amp = (max_amp-min_amp)*threshold+min_amp
        filter_index = abs(doppler_fft_data)<threshold_amp
        doppler_fft_data[filter_index] = 0

    return doppler_fft_data.transpose((2,1,3,0))

def show_rdp_phase(filename,slides=cfg.mydca.no_chirp,threshold=0):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating range doppler phase profile')
    # return 1-2,4-3,4-1,3-2,4-2,3-1
    # calc range doppler profile
    raw_data = np.load(filename)
    data = fmcw.int16_complex(raw_data)
    range_fft_data,_ = fmcw.range_fft(data)
    num_chirp = cfg.mydca.no_chirp
    num_frame = int((range_fft_data.shape[-1]-num_chirp)/slides) + 1
    doppler_fft_data = np.zeros((num_frame,range_fft_data.shape[0],range_fft_data.shape[1],num_chirp),dtype=complex)
    for i in range(num_frame):
        doppler_fft_data[i] = fmcw.doppler_fft(range_fft_data[:,:,i*slides:i*slides+num_chirp])

    # filter < threshold
    max_amp = abs(doppler_fft_data).max()
    min_amp = abs(doppler_fft_data).min()
    threshold_amp = (max_amp-min_amp)*threshold+min_amp
    filter_index = abs(doppler_fft_data)>threshold_amp
    doppler_fft_data[filter_index] = 0

    # calc doppler fft phase diff
    angle_doppler_fft_data = np.zeros(doppler_fft_data.shape)
    angle_doppler_fft_data[:,:,0,:] = np.angle(doppler_fft_data[:,:,0,:])-np.angle(doppler_fft_data[:,:,1,:])
    angle_doppler_fft_data[:,:,1,:] = np.angle(doppler_fft_data[:,:,3,:])-np.angle(doppler_fft_data[:,:,2,:])
    angle_doppler_fft_data[:,:,2,:] = np.angle(doppler_fft_data[:,:,3,:])-np.angle(doppler_fft_data[:,:,0,:])
    angle_doppler_fft_data[:,:,3,:] = np.angle(doppler_fft_data[:,:,2,:])-np.angle(doppler_fft_data[:,:,1,:])
    angle_doppler_fft_data[:,:,4,:] = np.angle(doppler_fft_data[:,:,3,:])-np.angle(doppler_fft_data[:,:,1,:])
    angle_doppler_fft_data[:,:,5,:] = np.angle(doppler_fft_data[:,:,2,:])-np.angle(doppler_fft_data[:,:,0,:])
    angle_doppler_fft_data -= (angle_doppler_fft_data>=np.pi)*np.pi*2
    angle_doppler_fft_data += (angle_doppler_fft_data<=-np.pi)*np.pi*2
    return angle_doppler_fft_data.transpose((2,1,3,0))

def show_rp_ssm(filename,interval=20,num_label=64):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating range self-similar matrix profile')
    # calc range profile
    raw_data = np.load(filename)
    data = fmcw.int16_complex(raw_data)
    range_fft_data,_ = fmcw.range_fft(data)
    range_fft_data = abs(range_fft_data[:,0,::interval])
    range_fft_data = fmcw.calc_pd(range_fft_data,num_label)

    ssm = np.zeros((range_fft_data.shape[-1],range_fft_data.shape[-1]))
    for i in range(range_fft_data.shape[-1]):
        for k in range(range_fft_data.shape[-1]):
            ssm[i,k] = metrics.normalized_mutual_info_score(range_fft_data[:,i],range_fft_data[:,k])

    return ssm[np.newaxis,:,:,np.newaxis]

def show_rdp_ssm(filename,slides=cfg.mydca.no_chirp,interval=20,num_label=64):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating range doppler self-similar matrix profile')
    # calc range doppler profile
    raw_data = np.load(filename)
    data = fmcw.int16_complex(raw_data)
    range_fft_data,_ = fmcw.range_fft(data)
    num_chirp = cfg.mydca.no_chirp
    num_frame = int((range_fft_data.shape[-1]-num_chirp)/slides) + 1
    doppler_fft_data = np.zeros((num_frame,range_fft_data.shape[0],range_fft_data.shape[1],num_chirp),dtype=complex)
    for i in range(num_frame):
        doppler_fft_data[i] = fmcw.doppler_fft(range_fft_data[:,:,i*slides:i*slides+num_chirp])

    doppler_fft_data = doppler_fft_data[:,:,0,:]
    doppler_fft_data = doppler_fft_data.reshape(doppler_fft_data.shape[0],-1)
    doppler_fft_data = doppler_fft_data[:,::interval]
    range_doppler_vector = fmcw.calc_pd(doppler_fft_data,num_label)
    range_doppler_vector = abs(range_doppler_vector)

    ssm = np.zeros((range_doppler_vector.shape[0],range_doppler_vector.shape[0]))
    for i in range(range_doppler_vector.shape[0]):
        for k in range(range_doppler_vector.shape[0]):
            ssm[i,k] = metrics.normalized_mutual_info_score(range_doppler_vector[i,:],range_doppler_vector[k,:])
    for i in range(range_doppler_vector.shape[0]):
        ssm[i,i] = ssm.mean()

    return ssm[np.newaxis,:,:,np.newaxis]

def show_mdp(filename,slides=cfg.mydca.no_chirp,remove_static=1,toprint=1):
    if(toprint==1):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating micro doppler profile')
    # calc range doppler profile
    doppler_fft_data = show_rdp(filename,slides=slides,remove_static=remove_static,toprint=0)[0]
    doppler_fft_data = abs(doppler_fft_data).sum(axis=0)
    mdp = doppler_fft_data[np.newaxis,:,:,np.newaxis]
    return mdp

def partial_mdp(range_doppler_profile,partial_min=0,partial_max=20):
    data = range_doppler_profile[:,partial_min:partial_max,:,:]
    data = abs(data).sum(axis=1)
    return data[:,:,:,np.newaxis]

def music(filename,slides=cfg.mydca.no_chirp,N=4,M=3,X=1):
    data = np.load(filename)
    if(X==1):
        data = data[:,[7,6,11,10],:]
    num_chirp = cfg.mydca.no_chirp
    pmusic = np.zeros((int((data.shape[0]-num_chirp)/slides),180))
    for k in range(int((data.shape[0]-num_chirp)/slides)):
        Rx = 0+0j
        curdata = data[k*slides:k*slides+num_chirp,:,:]
        for i in range(curdata.shape[0]):
            Rx += 1/curdata.shape[0] * np.dot(curdata[i],curdata[i].T.conjugate())
        A,_,_ = np.linalg.svd(Rx)
        d = np.linspace(0,0.5*(N-1),N)

        for i in range(180):
            phim = (i-90)/180*np.pi
            a = np.exp(-1j*2*np.pi*d*np.sin(phim))
            enta = np.dot(A[:,M:N].T.conjugate(),a)
            pmusic[k][i] = 1/abs(np.dot(enta.T.conjugate(),enta))
    return pmusic[np.newaxis,:,:,np.newaxis]

def hampel(X):
    length = X.shape[0] - 1
    k = 5
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）
 
    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf

def hampel_filter(data):
    for i in range(data.shape[2]):
        data[0,:,i,0] = hampel(data[0,:,i,0])
    return data

def remove_doppler_speed0(data,hampel_filt=1,zero_interpolation=0):
    range_doppler_profile = data[:,int(data.shape[1]/2)-4:int(data.shape[1]/2)+4,:,:]
    speed0 = int(range_doppler_profile.shape[1]/2)
    inter0 = (range_doppler_profile[:,speed0+3,:,:] - range_doppler_profile[:,speed0-3,:,:])
    if(zero_interpolation!=1):
        range_doppler_profile[:,speed0-2,:,:] = range_doppler_profile[:,speed0-3,:,:] + inter0*1/6
        range_doppler_profile[:,speed0-1,:,:] = range_doppler_profile[:,speed0-3,:,:] + inter0*2/6
        range_doppler_profile[:,speed0,:,:] = range_doppler_profile[:,speed0-3,:,:] + inter0*3/6
        range_doppler_profile[:,speed0+1,:,:] = range_doppler_profile[:,speed0-3,:,:] + inter0*4/6
        range_doppler_profile[:,speed0+2,:,:] = range_doppler_profile[:,speed0-3,:,:] + inter0*5/6
    else:
        min_data = min(data.reshape(-1))
        range_doppler_profile[:,speed0-2,:,:] = min_data
        range_doppler_profile[:,speed0-1,:,:] = min_data
        range_doppler_profile[:,speed0,:,:] = min_data
        range_doppler_profile[:,speed0+1,:,:] = min_data
        range_doppler_profile[:,speed0+2,:,:] = min_data
    
    if(hampel_filt==1):
        range_doppler_profile = hampel_filter(range_doppler_profile)
    data[:,int(data.shape[1]/2)-4:int(data.shape[1]/2)+4,:,:] = range_doppler_profile
    return data

def mdp_percentile(data,threshold_low=0.5):
    percentile_data = np.zeros_like(data)
    for j in range(percentile_data.shape[2]):
        all_val = data[:,:,j,:].reshape(-1).sum()
        for i in range(percentile_data.shape[1]):
            percentile_data[0,i,j,0] = data[0,i:,j,0].sum() / all_val
    mask = np.zeros_like(percentile_data)
    percentile_data = percentile_data > threshold_low
    # sr.showing(percentile_data)
    for i in range(percentile_data.shape[2]):
        for j in range(1,percentile_data.shape[1]):
            if(percentile_data[0,j,i,0]!=percentile_data[0,j-1,i,0]):
                mask[0,j,i,0] = 1
    return mask

def show_nearest_range_mdp(filename,threshold=1.1,win_slides=cfg.mydca.no_chirp,\
    min_range_slides=10,a=0,offset=5):

    data = show_rdp(filename,slides=win_slides)
    data_range = abs(data).sum(2)
    data_range = data_range[:,:,:,np.newaxis]

    # alldata = []
    # for i in range(10):
    #     curdata = partial_mdp(data,i*10,(i+1)*10)
    #     alldata.append(curdata[0,:,:,0])
    # sr.showing(alldata)

    # sr.showing(data_range)                  
    data_range_edge = np.zeros_like(data_range)
    for j in range(data_range.shape[2]):
        for i in range(data_range.shape[1]-1):
            cur_threshold = threshold * data_range[0,:,j,0].mean()
            if((data_range[0,i+1,j,0]-data_range[0,i,j,0])>cur_threshold):
                data_range_edge[0,i+1,j,0] = 1
            else:
                data_range_edge[0,i+1,j,0] = 0
    # sr.showing(data_range_edge)

    body_range = np.zeros((data_range_edge.shape[2]))
    for i in range(data_range.shape[2]):
        for j in range(data_range.shape[1]):
            if(data_range_edge[0,j,i,0]==0):
                pass
            else:
                body_range[i] = j
                break
    min_body_range = np.zeros_like(body_range,dtype=np.int32)
    for i in range(body_range.shape[0]-min_range_slides):
        min_body_range[i] = body_range[i:i+min_range_slides].min()
    
    min_body_range[-min_range_slides:] = min_body_range[-min_range_slides-1]
    # print(min_body_range)
    # print(data.shape,min_body_range.shape)
    # data = abs(data).sum(1)[:,:,:,np.newaxis]
    mdp_data = np.zeros((1,data.shape[2],data.shape[3],1))
    for i in range(mdp_data.shape[2]):
        mdp_data[0,:,i,0] = abs(data[0,min_body_range[i]+a:min_body_range[i]+a+offset,:,i]).sum(0)
        
    # sr.showing([abs(data[0,:,:,:]).sum(0),mdp_data[0,:,:,0]],filename)
    return mdp_data

def show_peak_range_mdp(filename,threshold=1.1,win_slides=cfg.mydca.no_chirp,\
    min_range_slides=10,a=-5,offset=10):
    data = show_rdp(filename,slides=32)
    # sr.showing(data)
    data_range = abs(data).sum(2)
    data_range = data_range[:,:,:,np.newaxis]

    # sr.showing(data_range)       
    min_body_range = np.zeros((data_range.shape[2]),dtype=np.int32)
    # print(data_range.shape)
    for j in range(data_range.shape[2]):
        max_flag = 0
        for i in range(data_range.shape[1]):
            if(data_range[0,i,j,0]>max_flag):
                max_flag = data_range[0,i,j,0]
                min_body_range[j] = i

    # print(min_body_range)
    # print(data.shape,min_body_range.shape)
    # data = abs(data).sum(1)[:,:,:,np.newaxis]

    mdp_data = np.zeros((1,data.shape[2],data.shape[3],1))
    for i in range(mdp_data.shape[2]):
        mdp_data[0,:,i,0] = abs(data[0,min_body_range[i]+a:min_body_range[i]+a+offset,:,i]).sum(0)
        
    # sr.showing(abs(data[0,:,:,:]).sum(0)[np.newaxis,:,:,np.newaxis]-mdp_data[:,:,:,:],filename)
    # sr.showing([abs(data[0,:,:,:]).sum(0),mdp_data[0,:,:,0],abs(data[0,:,:,:]).sum(0)-mdp_data[0,:,:,0]],filename)
    return mdp_data
    
def tracking_speed(data,threshold=8,slides=cfg.mydca.no_chirp):
    percent = []
    tracking = np.zeros((1,data.shape[1],data.shape[2],data.shape[3]))
    predicting = np.zeros((1,data.shape[1],data.shape[2],data.shape[3]))
    time_interval = slides * cfg.mydca.preriodicity
    range_bin = cfg.mydca.range_bin
    doppler_bin = cfg.mydca.doppler_bin
    speed0_bin = int(cfg.mydca.no_chirp/2)
    for i in range(tracking.shape[3]):
        tracking[0,:,:,i] = data[0,:,:,i] > data[0,:,:,i].reshape(-1).mean()*threshold
        correct_predict = 0
        if(i>10):
            for j in range(tracking.shape[1]):
                for k in range(tracking.shape[2]):
                    if(tracking[0,j,k,i-1]!=0):
                        cur_range = j * range_bin
                        cur_velocity = (k-speed0_bin) * doppler_bin
                        predict_range = cur_range + cur_velocity*time_interval
                        predict_range_index = int(predict_range/range_bin)
                        if(predict_range_index>=0):
                            predicting[0,predict_range_index,k,i] = 1
                    if(predicting[0,j,k,i]==tracking[0,j,k,i] and tracking[0,j,k,i]==1):
                        correct_predict += 1

        title1 = 'original-rdp'
        title2 = 'energy=' \
            + str(int((data[0,:,:,i] * tracking[0,:,:,i]).reshape(-1).sum()/(data[0,:,:,i].reshape(-1).sum())*100)) + '%'
        title3 = 'correct predicion=' + str(int(correct_predict/(tracking[0,:,:,i].reshape(-1).sum())*100)) + '%'
        percent.append([title1,title2,title3])
    
    return tracking+0.0001,predicting+0.0001,percent

def show_recover_acc(filename):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating recovered acc')
    data = np.load(filename)
    data = data.transpose((1,0))
    acc_data = data[0:3,:]
    angle_data = data[6:9,:]
    re_acc_data = np.zeros_like(acc_data)
    angle_data = angle_data / 180 * np.pi
    for i in range(acc_data.shape[1]):
        tx = np.array([[1,0,0],[0,np.cos(angle_data[0][i]),np.sin(angle_data[0][i])],\
            [0,-np.sin(angle_data[0][i]),np.cos(angle_data[0][i])]])
        ty = np.array([[np.cos(angle_data[1][i]),0,-np.sin(angle_data[1][i])],[0,1,0],\
            [np.sin(angle_data[1][i]),0,np.cos(angle_data[1][i])]])
        tz = np.array([[np.cos(angle_data[2][i]),np.sin(angle_data[2][i]),0],\
            [-np.sin(angle_data[2][i]),np.cos(angle_data[2][i]),0],[0,0,1]])
        temp = np.dot(tx,ty,tz)
        txyz_inv = np.linalg.inv(temp)
        axyz = np.array([acc_data[0][i],acc_data[1][i],acc_data[2][i]])
        xyz = np.dot(txyz_inv,axyz)
        re_acc_data[0][i] = xyz[0]
        re_acc_data[1][i] = xyz[1]
        re_acc_data[2][i] = xyz[2]

    return re_acc_data

def show_range_aoa(data_name):
    # using beamforming to calculate
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating range_aoa profile')
    rdp = show_rp(data_name,remove_static=1,toprint=0)
    rdp = rdp[[0,1,2,3],:,:,:]
    num_angle = 180
    # rdp = rdp[:,:,::40,:]
    ra = np.zeros((1,rdp.shape[1],num_angle,rdp.shape[2]),dtype=np.complex128)
    offset = np.zeros((num_angle,8),dtype=np.complex128)
    for angle in range(num_angle):
        curangle = angle - 90
        offset[angle,:] = np.array([np.exp(0j*np.pi*np.sin(curangle/180*np.pi)),\
            np.exp(1j*1*np.pi*np.sin(curangle/180*np.pi)),\
            np.exp(1j*2*np.pi*np.sin(curangle/180*np.pi)),\
            np.exp(1j*3*np.pi*np.sin(curangle/180*np.pi)),\
            np.exp(1j*4*np.pi*np.sin(curangle/180*np.pi)),\
            np.exp(1j*5*np.pi*np.sin(curangle/180*np.pi)),\
            np.exp(1j*6*np.pi*np.sin(curangle/180*np.pi)),\
            np.exp(1j*7*np.pi*np.sin(curangle/180*np.pi))])
    offset = offset[:,:rdp.shape[0]]
    for i in range(ra.shape[3]):
        ra[0,:,:,i] = np.matmul(offset,rdp[:,:,i,0]).transpose((1,0))
    return ra

def show_CVD(data_name,slides=cfg.mydca.no_chirp,recov=0):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'calculating cadence velocity diagram')
    mdp = show_mdp(data_name,slides=slides,toprint=0,recov=recov)
    cvd = fmcw.fft(mdp,axis=-2)
    cfg.mydca.time_bin_list = np.round(np.linspace(0,int(cvd.shape[2]/cfg.mydca.time_length),cvd.shape[2],endpoint=False),2)
    cfg.mydca.time_bin_list = cfg.mydca.time_bin_list[:int(cvd.shape[2]/2)]
    cvd = cvd[:,:,:int(cvd.shape[2]/2),:]
    return cvd

def recov_mdp(data):
    data = data[0,:,:,0]
    max_index = np.zeros((data.shape[1]),dtype=np.int32)
    for i in range(data.shape[1]):
        max_index[i] = np.argmax(data[:,i])
    for i in range(data.shape[1]-2):
        i += 1
        max_index[i] = int(max_index[i-1:i+2].mean())
    for i in range(data.shape[1]):
        a = np.hstack((data[:,i],data[:,i],data[:,i]))
        data[:,i] = a[(max_index[i]-int(data.shape[0]/2)+data.shape[0]):(max_index[i]+int(data.shape[0]/2)+data.shape[0])]
    return data[np.newaxis,:,:,np.newaxis]

def normal_mdp(data,threshold=0.05):
    # normalize mdp
    for i in range(data.shape[2]):
        min_val = data[0,:,i,0].min()
        max_val = data[0,:,i,0].max()
        data[0,:,i,0] = (data[0,:,i,0] - min_val)/(max_val-min_val)
    
    # gaussian low pass filter
    filter_size = 3
    data = cv2.GaussianBlur(data[0,:,:,0],(filter_size,filter_size),sigmaX=1)
    data = data[np.newaxis,:,:,np.newaxis]

    # remove value less than threshold
    mask = data > threshold
    data = data * mask
    
    return data

def get_mask(data_name,slides=cfg.mydca.no_chirp):
    mdp = show_mdp(data_name,slides=slides,toprint=0)
    mdp = normal_mdp(mdp)
    # calculate torso mask
    mask = np.zeros_like(mdp)
    for i in range(mdp.shape[2]):
        cur  = mdp[0,:,i,0]
        sort_cur = np.sort(cur)
        for j in range(3):
            cur_index = np.where(cur==sort_cur[-j-1])
            mask[0,cur_index[0][0],i,0] = 1
    # calculate limb mask
    mask1 = mdp_percentile(mdp,threshold_low=0.95)
    mask2 = mdp_percentile(mdp,threshold_low=0.75)
    mask3 = mdp_percentile(mdp,threshold_low=0.30)

    mask = mask + 10*mask1 + 100*mask2 + 1000*mask3
    return mask

def show_mdp_range(data_name,slides=cfg.mydca.no_chirp):
    mdp = show_mdp(data_name,slides=slides,toprint=0)
    mdp = normal_mdp(mdp,threshold=0.05)
    mask = get_mask(data_name,slides=slides)
    rdp = show_rdp(data_name,slides=slides,toprint=0)
    rtp = np.zeros((rdp.shape[2],rdp.shape[3]))
    rtp = rtp - 100
    print(mdp.shape,rdp.shape,mask.shape,rtp.shape)
    torso_index = np.zeros((mask.shape[2]))
    for i in range(torso_index.shape[0]):
        cur = mask[0,:,i,0]
        sort_cur = np.sort(cur)
        cur_index = np.where(cur==sort_cur[-1])[0][0]
        cur = rdp[0,:,cur_index,i]
        sort_cur = np.sort(cur)
        cur_index = np.where(cur==sort_cur[-1])[0][0]
        torso_index[i] = cur_index
    for i in range(rtp.shape[0]):
        for j in range(rtp.shape[1]):
            cur = abs(rdp[0,:,i,j])
            sort_cur = np.sort(cur)
            cur_index = np.where(cur==sort_cur[-1])[0][-1]
            if(mdp[0,i,j,0]!=0):
                rtp[i,j] = cur_index - torso_index[i]
    # print(torso_index[850],rtp[:,850])
    # plt.plot(torso_index)

    return rtp[np.newaxis,:,:,np.newaxis]

        


if __name__=='__main__':
    # data_name = '/Users/wenhao/Desktop/mmwave_data/rawdata/220520-hospital/22-05-20 08-48-18/dca.npy'
    data_name1 = '/Users/wenhao/Desktop/mmwave_data/rawdata/220523/22-05-23 15-58-03/dca.npy'
    data_name2 = '/Users/wenhao/Desktop/mmwave_data/rawdata/220520-hospital/22-05-20 09-40-34/dca.npy'
    
    # data_name1 = '/Users/wenhao/Desktop/mmwave_data/rawdata/220520-hospital/22-05-20 12-05-40/dca.npy'
    mask = get_mask(data_name1,slides=32)
    mdp = show_mdp(data_name1,slides=32)
    mdp = normal_mdp(mdp,threshold=0)
    sr.showing([mdp,mask+0.0001])

    # rtp = show_mdp_range(data_name1,slides=32)
    # # sr.showing(np.exp(rtp))
    # sns.heatmap(rtp[0,:,:,0],cmap='jet')
    # plt.show()

    # rdp = show_rdp(data_name1,slides=32)
    # rtp = abs(rdp[0,:,:,:]).sum(1)
    # rtp = rtp[np.newaxis,:,:,np.newaxis]
    # sr.showing(rtp)

    # rdp = show_rdp(data_name1,slides=32)
    # sr.showing(rdp)


'''
'''

    
