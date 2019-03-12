import numpy as np
import os
import shutil
import wfdb
import keras
from keras.models import load_model
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import os.path
import tensorflow as tf
from load_samples import load_samples

def predict_ecg(record_name,model,delay=0):
    # loads 10 seconds of the target ECG, predicts and returns result
    record = wfdb.rdrecord(record_name, pb_dir='staffiii/data/')
    x=np.zeros((10,9000,1))
    for i in range(10):
        x[i]=record.p_signal[(delay*1000)+(i*1000):(delay*1000)+((i+1)*1000),:].reshape(9000,1)
    y=np.sum(model.predict(x))/10
    return y

def evaluate_model_on_examples(threshold=0.5):
    samples=load_samples()

    model=load_model('model_0.h5')
    next_model_number=1

    y_pred=np.zeros((152,1))
    y_true=np.zeros((152,1))
    y_counter=0

    csv='ECG,Prediction,Truth,Correct'

    for i in range(len(samples)):
        s=samples[i]
        # change models after each hold-out set
        # this ensures each model is evaluated on patients whose data it has never encountered
        if (i%15==0 and next_model_number<5 and i!=0):
            model=load_model('model_'+str(next_model_number)+'.h5')
            print('Switching to model '+str(next_model_number)+' at patient '+str(i))
            next_model_number+=1
        # predict baseline ECG
        y_non_ischaemic=predict_ecg(str(s[0]).zfill(3)+'a',model,0)
        y_pred[y_counter]=y_non_ischaemic
        y_true[y_counter]=0
        y_counter+=1
        y_non_ischaemic=int((y_non_ischaemic>threshold)*1)
        csv+='\n'+str(s[0]).zfill(3)+'a,'+str(y_non_ischaemic)+',0,'
        if y_non_ischaemic==0:
            csv+='1'
        else:
            csv+='0'
        # predict balloon inflation ECG
        y_ischaemic=predict_ecg(str(s[0]).zfill(3)+s[4],model,int(s[1])+60)
        y_pred[y_counter]=y_ischaemic
        y_true[y_counter]=1
        y_counter+=1
        y_ischaemic=int((y_ischaemic>threshold)*1)
        csv+='\n'+str(s[0]).zfill(3)+s[4]+','+str(y_ischaemic)+',1,'
        if y_ischaemic==1:
            csv+='1'
        else:
            csv+='0'

    # write results to file
    f=open('db_results.csv','w')
    f.write(csv)
    f.close

    # save results as arrays for threshold search
    np.save('y_true.npy',y_true)
    np.save('y_pred.npy',y_pred)

if __name__=="__main__":
    evaluate_model_on_examples()