from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import math


def func(data,sample_rate=50,accumulated=1):
    
    for i in range(len(data)):
        print(len(data[i]))
        data[i] = np.array(data[i][:600])
        
    # get the angular speed at every sample
    for i in range(3,6):
        data[i][0] = data[i][0] / sample_rate
        for j in range(data[i].shape[0]-1):
            data[i][j+1] = data[i][j+1]/sample_rate + data[i][j]
    # using the rotation angle to recover acc
    [ax,ay,az,rx,ry,rz] = data
    for i in range(ax.shape[0]):
        cur = ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]
        tx = np.array([[1,0,0],[0,np.cos(rx[i]),np.sin(rx[i])],[0,-np.sin(rx[i]),np.cos(rx[i])]])
        ty = np.array([[np.cos(ry[i]),0,-np.sin(ry[i])],[0,1,0],[np.sin(ry[i]),0,np.cos(ry[i])]])
        tz = np.array([[np.cos(rz[i]),np.sin(rz[i]),0],[-np.sin(rz[i]),np.cos(rz[i]),0],[0,0,1]])
        temp = np.dot(tx,ty)
        temp = np.dot(temp,tz)
        txyz_inv = np.linalg.inv(temp)
        axyz = np.array([ax[i],ay[i],az[i]])
        xyz = np.dot(txyz_inv,axyz)
        data[0][i] = xyz[0]
        data[1][i] = xyz[1]
        data[2][i] = xyz[2]
    if(accumulated==1):
        data[2] = data[2] - 9.8
        # acc_data = data.copy()
        # cur_v = np.zeros(3)
        # for j in range(data[0].shape[0]-1):
        #     for i in range(3):
        #         cur_v[i] += acc_data[i][j]
        #         data[i][j+1] = 0.5*acc_data[i][j]*pow((1/sample_rate),2) + cur_v[i]/sample_rate
        data[0][0] /= sample_rate
        data[1][0] /= sample_rate
        data[2][0] /= sample_rate
        
        for j in range(data[0].shape[0]-1):
            for i in range(3):
                data[i][j+1] = data[i][j+1] / sample_rate + data[i][j]
    return data

def showing(data):
    num = data.shape[0]
    for i in range(num):
        ax1 = plt.subplot(num,1,i+1)
        ax1.plot(data[i,:])
    plt.show()

def recover_acc(acc_data,angle_data):
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
        print(xyz)
        re_acc_data[0][i] = xyz[0]
        re_acc_data[1][i] = xyz[1]
        re_acc_data[2][i] = xyz[2]

    return re_acc_data

    


if __name__=='__main__':
    dirname = 'C:/Users/timmothy/Desktop/Matlab_usb_data/npy_data/'
    filename = '2022_5_18-13_46_24.npy'
    data = np.load(dirname+filename)
    data = data.transpose((1,0))
    print(data.shape)
    data = recover_acc(data[0:3],data[6:9])
    showing(data)
    




