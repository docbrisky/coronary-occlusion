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

def load_pretrained_model():    
    # loads model pretrained on arrhythmia detection task (34 layer CNN with 4D output)
    # swaps last layer for a 1D output with sigmoid activation

    model = load_model('pretrained-arrhythmia-model.hdf5')
    dense=Dense(1,activation='sigmoid')(model.layers[-2].output)
    model=Model(inputs=model.input,outputs=dense)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    return model

def populate_X_Y(X,Y,samples,non_ischaemic_seconds,ischaemic_seconds):
    # downloads ECG records from Physionet and generates input vectors / class labels

    placeholder=0
    for s in samples:
        record_name=str(s[0]).zfill(3)
        record = wfdb.rdrecord(record_name+'a', pb_dir='staffiii/data/')
        for i in range(non_ischaemic_seconds*10):
            ecg_slice=record.p_signal[i*100:(i*100)+1000,:]
            x=ecg_slice.reshape((9000,1))
            X[placeholder]=x
            Y[placeholder]=0
            placeholder+=1
        record = wfdb.rdrecord(record_name+s[4], pb_dir='staffiii/data/')
        start_position=int(s[1])*1000 + 60000
        for i in range(ischaemic_seconds*10):
            ecg_slice=record.p_signal[(i*100)+start_position:(i*100)+1000+start_position,:]
            x=ecg_slice.reshape((9000,1))
            X[placeholder]=x
            Y[placeholder]=1
            placeholder+=1
        print('Done with record',samples.index(s),'of',len(samples),end="\r")
    return X,Y

def train_model(model,X,Y,X_val,Y_val,model_no):
    # trains model
    # note class weights overcompensate for class distribution
    # clinically, a false negative is worst outcome, so model incurs greater loss for this during training

    checkpointer=ModelCheckpoint('model_'+str(model_no)+'.h5',verbose=1,save_best_only=True)
    model.fit(x=X,y=Y,
              batch_size=64,
              validation_data=(X_val,Y_val),
              epochs=3,
              callbacks=[checkpointer],
              class_weight = {0: 1.,1: 8.}
              )

def evaluate_model(model,X_val,Y_val,model_no):
    # evaluates model on hold-out validation set
    # true positives, true negatives, etc. initiated at 10e-5 to avoid division by zero in later calculations

    Y_pred=model.predict(X_val,batch_size=64)

    tp=0.00001
    tn=0.00001
    fp=0.00001
    fn=0.00001

    for i in range(X_val.shape[0]//2100):
        y_val_neg=np.sum(Y_val[i*2100:(i*2100)+1800,:])/1800
        y_val_pos=np.sum(Y_val[(i*2100)+1800:(i+1)*2100,:])/300

        if (y_val_neg==0) and (y_val_pos==1):
            y_pred_neg=np.sum(Y_pred[i*2100:(i*2100)+1800,:])/1800
            y_pred_pos=np.sum(Y_pred[(i*2100)+1800:(i+1)*2100,:])/300

            if round(y_pred_neg)==0:
                tn+=1
            else:
                fp+=1
            if round(y_pred_pos)==1:
                tp+=1
            else:
                fn+=1
        
        else:
            print('Results error!')

    sens=tp/(tp+fn)
    spec=tn/(tn+fp)
    ppv=tp/(tp+fp)

    results='Sensitivity: '+str(sens)+' Specificity: '+str(spec)+' PPV: '+str(ppv)
    print(results)
    f=open('Results_'+str(model_no)+'.txt','w')
    f.write(results)
    f.close()
    
if __name__ == "__main__":
    # this script represents an initial feasibility experiment, hence only limited data are used:
    non_ischaemic_seconds=180
    ischaemic_seconds=30

    samples=load_samples()

    # 180 ischaemic seconds + 30 non-ischaemic seconds = 210 seconds per patient
    # data acquired using sliding windows with 100mS lateral shifts
    # hence, 10 windows per second = 2100 windows per patient
    # data is subsequently augmented by flipping voltage of each window, so 2 x 2100 = 4200 windows per patient
    X=np.zeros((len(samples)*4200,9000,1))
    Y=np.zeros((len(samples)*4200,1))

    X,Y=populate_X_Y(X,Y,samples,non_ischaemic_seconds,ischaemic_seconds)

    for i in range(len(samples)):
        # saves all the data to local disk (no augmentation at this stage, so 2100 windows / patient)
        x_neg=X[i*2100:(i*2100)+1800,:,:]
        x_pos=X[(i*2100)+1800:(i+1)*2100:,:]
        np.save('Patients/'+str(i).zfill(2)+'_pos.npy',x_pos)
        np.save('Patients/'+str(i).zfill(2)+'_neg.npy',x_neg)

    for i in range(5):
        # load the model pretrained on arrhythmia task
        model=load_pretrained_model()

        # records for 77 patients included in study (there is a duplicate record, so actually 76 patients in write-up)
        # first four hold-out sets have 15 patients each, so require 15 loops through data
        # final hold-out set has 17 patients, so required 17 loops and is coded further below
        if i <4:
            X_train=np.zeros((62*4200,9000,1))
            Y_train=np.zeros((62*4200,1))
            X_val=np.zeros((15*4200,9000,1))
            Y_val=np.zeros((15*4200,1))
            # add data up to start of validation set to training set:
            for h in range(i*15):
                # add non-augmented data to training set
                X_train[h*4200:(h*4200)+1800,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')
                Y_train[h*4200:(h*4200)+1800,:]=np.zeros((1800,1))
                X_train[(h*4200)+1800:(h*4200)+2100,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')
                Y_train[(h*4200)+1800:(h*4200)+2100,:]=np.ones((300,1))
                # add augmented data to training set (voltaged flipped)
                X_train[(h*4200)+2100:(h*4200)+3900,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')*-1
                Y_train[(h*4200)+2100:(h*4200)+3900,:]=np.zeros((1800,1))
                X_train[(h*4200)+3900:(h+1)*4200,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')*-1
                Y_train[(h*4200)+3900:(h+1)*4200,:]=np.ones((300,1))
            # add data to validation set:
            for h in range(i*15,(i+1)*15):
                j=h-(i*15)
                X_val[j*4200:(j*4200)+1800,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')
                Y_val[j*4200:(j*4200)+1800,:]=np.zeros((1800,1))
                X_val[(j*4200)+1800:(j*4200)+2100,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')
                Y_val[(j*4200)+1800:(j*4200)+2100,:]=np.ones((300,1))
                X_val[(j*4200)+2100:(j*4200)+3900,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')*-1
                Y_val[(j*4200)+2100:(j*4200)+3900,:]=np.zeros((1800,1))
                X_val[(j*4200)+3900:(j+1)*4200,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')*-1
                Y_val[(j*4200)+3900:(j+1)*4200,:]=np.ones((300,1))
            # add data from end of validation set to training set:
            for h in range((i+1)*15,76):
                j=h-((i+1)*15)
                X_train[j*4200:(j*4200)+1800,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')
                Y_train[j*4200:(j*4200)+1800,:]=np.zeros((1800,1))
                X_train[(j*4200)+1800:(j*4200)+2100,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')
                Y_train[(j*4200)+1800:(j*4200)+2100,:]=np.ones((300,1))
                X_train[(j*4200)+2100:(j*4200)+3900,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')*-1
                Y_train[(j*4200)+2100:(j*4200)+3900,:]=np.zeros((1800,1))
                X_train[(j*4200)+3900:(j+1)*4200,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')*-1
                Y_train[(j*4200)+3900:(j+1)*4200,:]=np.ones((300,1))
        # for final hold-out set of 16 patients
        elif i==4:
            X_train=np.zeros((60*4200,9000,1))
            Y_train=np.zeros((60*4200,1))
            X_val=np.zeros((17*4200,9000,1))
            Y_val=np.zeros((17*4200,1))
            for h in range(60):
                X_train[h*4200:(h*4200)+1800,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')
                Y_train[h*4200:(h*4200)+1800,:]=np.zeros((1800,1))
                X_train[(h*4200)+1800:(h*4200)+2100,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')
                Y_train[(h*4200)+1800:(h*4200)+2100,:]=np.ones((300,1))
                X_train[(h*4200)+2100:(h*4200)+3900,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')*-1
                Y_train[(h*4200)+2100:(h*4200)+3900,:]=np.zeros((1800,1))
                X_train[(h*4200)+3900:(h+1)*4200,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')*-1
                Y_train[(h*4200)+3900:(h+1)*4200,:]=np.ones((300,1))
            for h in range(60,76):
                j=h-60
                X_val[j*4200:(j*4200)+1800,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')
                Y_val[j*4200:(j*4200)+1800,:]=np.zeros((1800,1))
                X_val[(j*4200)+1800:(j*4200)+2100,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')
                Y_val[(j*4200)+1800:(j*4200)+2100,:]=np.ones((300,1))
                X_val[(j*4200)+2100:(j*4200)+3900,:,:]=np.load('Patients/'+str(h).zfill(2)+'_neg.npy')*-1
                Y_val[(j*4200)+2100:(j*4200)+3900,:]=np.zeros((1800,1))
                X_val[(j*4200)+3900:(j+1)*4200,:,:]=np.load('Patients/'+str(h).zfill(2)+'_pos.npy')*-1
                Y_val[(j*4200)+3900:(j+1)*4200,:]=np.ones((300,1))
    
        # train the model
        train_model(model,X_train,Y_train,X_val,Y_val,i)

        # finally, evaluate the model
        evaluate_model(model,X_val,Y_val,i)

        # note that the results of this evluation are preliminary and will not correspond to results in paper
        # fuller evaluation process undertaken later