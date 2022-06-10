from re import A

from matplotlib.axis import YAxis
import matplotlib
import common.config as cfg
from common import fmcw
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
import seaborn as sns
from sklearn import metrics
import gc
import signalprocess as sp

def showing(data,fig_title='',sub_title=None,xaxis=[],yaxis=[]):
    # print(data.shape)
    # data = remove_part(data,0.25)
    # data = remove_part(data,0.5)
    if(type(data)==list):
        if(data[0].shape[-1]==1):
            picnum = len(data)
            fig1 = plt.figure(fig_title,figsize=(15,6)) 
            titles = ['original mdp','partial mdp']
            for i in range(len(data)):
                curdata = data[i]
                ax1 = fig1.add_subplot(len(data),1,i+1)
                # plt.title(str(round(10*i,2))+'~'+str(round(10*(i+1),2)))
                # plt.title(titles[i])
                ax1.imshow(np.log(abs(curdata[0,:,:,0])),cmap=plt.cm.rainbow)
            plt.show()
        else:
            plt.ion()
            fig1 = plt.figure(fig_title,figsize=(9,6))
            toplot = min(data[0].shape[-1],data[1].shape[-1])
            for i in range(toplot):
                print(i,toplot)
                for j in range(len(data)):
                    ax1 = fig1.add_subplot(1,len(data),j+1)
                    if(sub_title!=None):
                        plt.title(sub_title[j])
                    im = ax1.imshow(np.log(abs(data[j][0,:,:,i])),cmap=plt.cm.rainbow)
                plt.pause(0.001)
                fig1.clf()
            plt.ioff()
    elif(len(data.shape)==1):
        plt.plot(data)
        plt.show()
    elif(data.shape[-1]==1):
        fig1 = plt.figure(fig_title,figsize=(10,10))
        ax1 = fig1.add_subplot(1,1,1)
        if(type(xaxis)!=list):
            ax1.set_xticks(np.linspace(0,data.shape[2],data.shape[2],endpoint=False)[::10],xaxis[::10],rotation='vertical')
        if(type(yaxis)!=list):
            ax1.set_yticks(np.linspace(0,data.shape[1],data.shape[1],endpoint=False)[::10],yaxis[::10])
        ax1.imshow(np.log(abs(data[0,:,:,0])),cmap=plt.cm.rainbow)
        
        plt.show()
    elif(data.shape[-1]!=1):
        plt.ion()
        fig1 = plt.figure(fig_title,figsize=(5,5))
        for i in range(data.shape[-1]):
            print(i,data.shape[-1])
            ax1 = fig1.add_subplot(1,1,1)
            if(type(sub_title)==str):
                ax1.set_title(sub_title)
            elif(type(sub_title)==list):
                ax1.set_title(sub_title[i])
            else:
                pass
            im = ax1.imshow(np.log(abs(data[0,:,:,i])),cmap=plt.cm.rainbow)
            plt.pause(0.001)
            fig1.clf()
        plt.ioff()

def func(data,low_per = 0.4,high_per = 1):
    data = sp.remove_doppler_speed0(data)
    data = sp.mdp_percentile(data,low_per,high_per)
    not_zero = data!=0
    is_zero = data==0

    avg_data = data.reshape(-1).sum()/not_zero.reshape(-1).sum()
    print(avg_data,data.reshape(-1).sum(),not_zero.reshape(-1).sum())
    data += avg_data*is_zero
    return data
        
def showing3D_heatmap(data):
    print(data.shape)
    x = np.zeros((data.shape[0]*data.shape[1]*data.shape[2]),dtype=np.int32)
    y = np.zeros((data.shape[0]*data.shape[1]*data.shape[2]),dtype=np.int32)
    z = np.zeros((data.shape[0]*data.shape[1]*data.shape[2]),dtype=np.int32)
    val = np.zeros((data.shape[0]*data.shape[1]*data.shape[2]))
    for i in range(data.shape[0]):
        print(i,data.shape[0])
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                x[i*data.shape[1]*data.shape[2]+j*data.shape[2]+k] = i
                y[i*data.shape[1]*data.shape[2]+j*data.shape[2]+k] = j
                z[i*data.shape[1]*data.shape[2]+j*data.shape[2]+k] = k*10
                val[i*data.shape[1]*data.shape[2]+j*data.shape[2]+k] = abs(data[i,j,k])
    min_v = min(val)
    max_v = max(val)
    # color = [plt.get_cmap("seismic", 100)(int(float(i-min_v)/(max_v-min_v)*100)) for i in val]

    # 2.0 显示三维散点图
    # 新建一个figure()
    fig = plt.figure()
    # 在figure()中增加一个subplot，并且返回axes
    ax = fig.add_subplot(111,projection='3d')
    # 设置colormap，与上面提到的类似，使用"seismic"类型的colormap，共100个级别
    # plt.set_cmap(plt.get_cmap("seismic", 100))
    # 绘制三维散点，各个点颜色使用color列表中的值，形状为"."
    im = ax.scatter(x, y, z, s=1,c=val,marker='.',alpha=0.5)
    # 2.1 增加侧边colorbar
    # 设置侧边colorbar，colorbar上显示的值使用lambda方程设置
    # fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_v-min_v)+min_v)))
    # 2.2 增加坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 2.3显示
    plt.show()



if __name__=='__main__':
    data_name1 = '/Users/wenhao/Desktop/mmwave_data/rawdata/220523/22-05-23 15-59-10/dca.npy'
    data_name2 = '/Users/wenhao/Desktop/mmwave_data/rawdata/220520-hospital/22-05-20 09-40-34/dca.npy'
    # mask = get_mask(data_name1,slides=32)
    # mdp = show_mdp(data_name1,slides=32)
    # mdp = normal_mdp(mdp,threshold=0.05)
    # sr.showing([mdp,mask])
    

