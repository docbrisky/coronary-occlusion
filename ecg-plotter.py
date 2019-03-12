import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import wfdb
import os.path

def load_samples():
    samples=[]
    exclusion_list=[1,4,5,6,89] #these records have erroneous lead placement
    f=open('timings.txt','r')
    txt=f.read()
    t_list=txt.split('\n')
    g=open('balloons.txt','r')
    balloon_txt=g.read()
    b_list=balloon_txt.split('\n')
    for t in t_list:
        if t!='X':
            if (t_list.index(t)+1 not in exclusion_list):
                t_split=t.split(';')
                if (int(t_split[2]))>90:
                    s=[]
                    s.append(t_list.index(t)+1)
                    s.append(t_split[0])
                    s.append(t_split[1])
                    s.append(t_split[2])
                    s.append(b_list[t_list.index(t)].split(str(t_list.index(t)+1))[-1])
                    samples.append(s)
    return samples

def plot_ecg(record_name,ischaemic=False,delay=0,duration=2500):
    record = wfdb.rdrecord(record_name, pb_dir='staffiii/data/')
    ecg_9=record.p_signal
    aVR=(ecg_9[(delay*1000)+duration:(delay*1000)+(duration*2),7]+ecg_9[(delay*1000)+duration:(delay*1000)+(duration*2),6])/-2
    aVL=ecg_9[(delay*1000)+duration:(delay*1000)+(duration*2),6]-(ecg_9[(delay*1000)+duration:(delay*1000)+(duration*2),7]/2)
    aVF=ecg_9[(delay*1000)+duration:(delay*1000)+(duration*2),7]-(ecg_9[(delay*1000)+duration:(delay*1000)+(duration*2),6]/2)
    ecg=np.zeros((duration,12))
    ecg[:,0]=ecg_9[(delay*1000):(delay*1000)+duration,6]
    ecg[:,1]=aVR
    ecg[:,2]=ecg_9[(delay*1000)+(duration*2):(delay*1000)+(duration*3),0]
    ecg[:,3]=ecg_9[(delay*1000)+(duration*3):(delay*1000)+(duration*4),3]
    ecg[:,4]=ecg_9[(delay*1000):(delay*1000)+duration,7]
    ecg[:,5]=aVL
    ecg[:,6]=ecg_9[(delay*1000)+(duration*2):(delay*1000)+(duration*3),1]
    ecg[:,7]=ecg_9[(delay*1000)+(duration*3):(delay*1000)+(duration*4),4]
    ecg[:,8]=ecg_9[(delay*1000):(delay*1000)+duration,8]
    ecg[:,9]=aVF
    ecg[:,10]=ecg_9[(delay*1000)+(duration*2):(delay*1000)+(duration*3),2]
    ecg[:,11]=ecg_9[(delay*1000)+(duration*3):(delay*1000)+(duration*4),5]
    leads=['I','aVR','V1','V4','II','aVL','V2','V5','III','aVF','V3','V6']
    fig, axs = plt.subplots(3, 4, figsize=(50,25))
    fig.tight_layout()
    axs=np.reshape(axs,(12))
    #print(axs[0].__dict__)
    lead_order=[6,8,0,3,7,9,1,4,8,10,2,5]
    for i in range(12): #lead_order:
        axs[i].plot(ecg[:,i],c='k',linewidth=1.0)
        #axes.set_xlim([xmin,xmax])
        axs[i].set_ylim([-2.5,2.5])
        for h in range(-25,25,1):
            x=np.array([0,2500])
            y=np.array([h/10,h/10])
            axs[i].plot(x,y,c='r',linewidth=0.2)
        for h in range(0,2500,40):
            x=np.array([h,h])
            y=np.array([-2.5,2.5])
            axs[i].plot(x,y,c='r',linewidth=0.2)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].set_title(leads[i],fontsize=20)
    folder='Non-ischaemic'
    if (ischaemic):
        folder='Ischaemic'
    fig.savefig(folder+'/'+record_name+'.png', bbox_inches='tight')
    plt.close()

if __name__=="__main__":
    samples=load_samples()

    for s in samples:
        plot_ecg(str(s[0]).zfill(3)+'a')
        plot_ecg(str(s[0]).zfill(3)+s[-1],True)