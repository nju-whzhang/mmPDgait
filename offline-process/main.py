import signalprocess as sp
import os
import numpy as np
import time
import common.config as cfg

def processing(dir2process,rootname='F:/2022/mmwave_data/rawdata/',outputname='F:/2022/mmwave_data/processeddata/',cur_slides=cfg.mydca.no_chirp):
    rootname = rootname + dir2process + '/'
    outputname = outputname + dir2process + '/'
    filenames = os.listdir(rootname)
    for i in range(len(filenames)):
        curname = filenames[i]
        cur_acc_data = None
        if(os.path.isdir(rootname+curname) and curname!='video'):
            curdir = rootname + curname + '/'
            curout = outputname + curname + '/'
            curdata = curdir+'dca.npy'
            for j in os.listdir(curdir):
                if(j[0:3]=='acc'):
                    cur_acc_data = curdir + j
            
            if not os.path.exists(curout):
                os.makedirs(curout)
            print(str(i)+'-'+str(len(filenames)),'data to be processed:',curdata)
            
            range_profile = sp.show_rp(curdata)
            np.save(curout+'range_profile.npy',range_profile)

            range_doppler_profile = sp.show_rdp(curdata,slides=cur_slides)
            np.save(curout+'range_doppler_profile.npy',range_doppler_profile)

            # range_doppler_phase_profile = sp.show_rdp_phase(curdata,slides=cur_slides)
            # np.save(curout+'range_doppler_phase_profile.npy',range_doppler_phase_profile)

            # range_profile_ssm_profile = sp.show_rp_ssm(curdata)
            # np.save(curout+'range_profile_ssm_profile.npy',range_profile_ssm_profile)

            # range_doppler_ssm_profile = sp.show_rdp_ssm(curdata,slides=cur_slides)
            # np.save(curout+'range_doppler_ssm_profile.npy',range_doppler_ssm_profile)

            micro_doppler_profile = sp.show_mdp(curdata,slides=cur_slides)
            np.save(curout+'micro_doppler_profile.npy',micro_doppler_profile)

            range_aoa_profile = sp.show_range_aoa(curdata)
            np.save(curout+'range_aoa_profile.npy',range_aoa_profile)

            if(cur_acc_data!=None):
                recovered_acc = sp.show_recover_acc(cur_acc_data)
                np.save(curout+'recovered_acc_data.npy',recovered_acc)
        
if __name__=='__main__':
    processing('220523',cur_slides=16)
    

