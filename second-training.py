import numpy as np
import os
import shutil
import wfdb
import keras
from keras.models import load_model
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import os.path
import tensorflow as tf
import glob
from random import shuffle
from keras.callbacks import Callback
from load_samples import load_samples

def save_data(file_path):
    samples=load_samples()
    patient=0
    validation_segment=0

    for i in range(len(samples)):
        if (i%15==0 and i!=0 and validation_segment<5):
            validation_segment+=1
        # load up the sample to be looped over
        s=samples[i]
        # set the filename for the baseline ECG
        baseline_record=str(s[0]).zfill(3)+'a'
        # and the balloon inflation ECG (referred to elsewhere as the ischaemic ECG)
        balloon_record=str(s[0]).zfill(3)+s[4]
        # all baseline ECGs were 300 seconds
        baseline_seconds=300
        # the first 60 seconds of balloon inflation were discarded
        balloon_start=int(s[1])+60
        balloon_seconds=int(s[2])-60
        # write baseline record:
        record = wfdb.rdrecord(baseline_record, pb_dir='staffiii/data/')
        ecg=record.p_signal
        filename=file_path+'0\\'+str(patient).zfill(2)+'.npy'
        np.save(filename,ecg)
        record = wfdb.rdrecord(balloon_record, pb_dir='staffiii/data/')
        ecg=record.p_signal
        # write balloon record:
        filename=file_path+'1\\'+str(patient).zfill(2)+'.npy'
        np.save(filename,ecg[(balloon_start*1000):(balloon_start*1000)+(balloon_seconds*1000)])
        print('Written '+str(patient+1)+' patient files of '+str(len(samples)))
        patient+=1

def evaluate_models(file_path,model_number=5,appendix=''):
    # note this evaluation is based on a decision threshold=0.5
    # (full results in paper are based on a decision threshold=0.67)
    # note also this is based on all ECG data
    # (full results in paper are based on two 10s ECGs per patient, for direct comparison with expert cardiologists)
    tn=0
    fn=0
    tp=0
    fp=0
    acc=0

    overall_y=np.zeros((154,2))
    y_counter=0

    # if model_number==5, loop over all models (models are numbered 0-4)
    if model_number==5:
        start=0
        stop=5
    # else just evaluate the model specified
    else:
        start=model_number
        stop=model_number+1

    for i in range(start,stop):
        file_list=[]
        model=load_model('model_retrained_'+str(i)+'.h5')
        excluded_list=[]
        # exclude all the patients that the model has seen before from the validation set
        for h in range(76):
            if i==4:
                if (i*15)<=h:
                    excluded_list.append(str(h).zfill(2))
            else:
                if (i*15)<=h<((i+1)*15):
                    excluded_list.append(str(h).zfill(2))
        # load the negative (baseline) samples and predict the class for each
        for filename in glob.iglob(file_path+'0\\*.npy'):
            patient=filename.split('\\')[-1].split('.')[0]
            if patient in excluded_list:
                ecg=np.load(filename)
                x_shape=ecg.shape[0]//1000
                X=np.zeros((x_shape,9000,1))
                for h in range(x_shape):
                    X[h]=ecg[h*1000:(h+1)*1000].reshape((9000,1))
                print('Loaded ',x_shape,' baseline windows for patient ',patient,'\nPredicting diagnosis...')
                y=model.predict(X,batch_size=128,verbose=1)
                y=np.sum(y)/y.shape[0]
                overall_y[y_counter,0]=y
                overall_y[y_counter,1]=0.
                y_counter+=1
                y=int((y>=0.5)*1)
                if y==0:
                    print('Diagnosis correct')
                    tn+=1
                    acc+=1
                else:
                    print('Diagnosis incorrect')
                    fn+=1
        # load the positive (balloon inflation) samples and predict the class for each
        for filename in glob.iglob(file_path+'1\\*.npy'):
            patient=filename.split('\\')[-1].split('.')[0]
            if patient in excluded_list:
                ecg=np.load(filename)
                x_shape=ecg.shape[0]//1000
                X=np.zeros((x_shape,9000,1))
                for h in range(x_shape):
                    X[h]=ecg[h*1000:(h+1)*1000].reshape((9000,1))
                print('Loaded ',x_shape,' balloon windows for patient ',patient,'\nPredicting diagnosis...')
                y=model.predict(X,batch_size=128,verbose=1)
                y=np.sum(y)/y.shape[0]
                overall_y[y_counter,0]=y
                overall_y[y_counter,1]=1.
                y_counter+=1
                y=int((y>=0.5)*1)
                if y==1:
                    print('Diagnosis correct')
                    tp+=1
                    acc+=1
                else:
                    print('Diagnosis incorrect')
                    fp+=1
    # calculate the metrics
    sens=tp/(tp+fn)
    spec=tn/(tn+fp)
    ppv=tp/(tp+fp)
    acc=acc/154
    # print and save the results
    summary='Sensitivity: '+str(sens)+' Specificity: '+str(spec)+' PPV: '+str(ppv)+' Acc: '+str(acc)
    print(summary)
    f=open('Summary_retrained_'+appendix+'.txt','w')
    f.write(summary)
    f.close()

    np.save('Overall_y.npy',overall_y)

def retrain_old_models_new_data(file_path):
    # note samples were acquired by sliding windows approach with 250mS lateral shifts during this training process
    # (100mS lateral shifts used during initial training)
    # rationale was that smaller lateral shifts will boost data more at the expense of increased similarity between samples
    # more data was used during this process, so it was felt that larger shifts could be afforded
    print('Training old models with new data...')

    for i in range(5):
        pos_examples=0
        neg_examples=0
        excluded_list=[]
        windows=[]

        # load the models trained during feasibility study (see 'first-training.py')
        # freeze all but the final two layers (one of which is a flattener layer with no trainable parameters)
        print('Loading model',i+1,'of 5...')
        model=load_model('model_'+str(i)+'.h5')
        for l in model.layers[:-2]:
            l.trainable=False
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
        print('Model loaded')

        # create list of patients in hold-out set:
        for h in range(76):
            if i < 5:
                if (i*15)<=h<((i+1)*15):
                    excluded_list.append(str(h).zfill(2))
            else:
                if (i*15)<=h:
                    excluded_list.append(str(h).zfill(2))

        print('The hold-out set for this model will consist of patients',excluded_list[0],'to',excluded_list[-1])

        # add negative samples to training data
        for filename in glob.iglob(file_path+'0\\*.npy'):
            patient=filename.split('\\')[-1].split('.')[0]
            if patient not in excluded_list:
                x=np.load(filename)
                seconds=x.shape[0]//1000
                for h in range(180,seconds):
                    window=[x[h*1000:(h+1)*1000].reshape((9000,1)),0.]
                    windows.append(window)
                for h in range(180,seconds-1):
                    window=[x[(h*1000)+250:((h+1)*1000)+250].reshape((9000,1))*-1,0.]
                    windows.append(window)
                for h in range(180,seconds-1):
                    window=[x[(h*1000)+500:((h+1)*1000)+500].reshape((9000,1)),0.]
                    windows.append(window)
                for h in range(180,seconds-1):
                    window=[x[(h*1000)+750:((h+1)*1000)+750].reshape((9000,1))*-1,0.]
                    windows.append(window)
                neg_examples+=seconds
        print('\n')
        # add positive samples to training data
        for filename in glob.iglob(file_path+'1\\*.npy'):
            patient=filename.split('\\')[-1].split('.')[0]
            if patient not in excluded_list:
                x=np.load(filename)
                seconds=x.shape[0]//1000
                for h in range(30,seconds):
                    window=[x[h*1000:(h+1)*1000].reshape((9000,1)),1.]
                    windows.append(window)
                for h in range(30,seconds-1):
                    window=[x[(h*1000)+250:((h+1)*1000)+250].reshape((9000,1))*-1,1.]
                    windows.append(window)
                for h in range(30,seconds-1):
                    window=[x[(h*1000)+500:((h+1)*1000)+500].reshape((9000,1)),1.]
                    windows.append(window)
                for h in range(30,seconds-1):
                    window=[x[(h*1000)+750:((h+1)*1000)+750].reshape((9000,1))*-1,1.]
                pos_examples+=seconds
        print('\nLoaded',len(windows),'one-second training windows for training set',i+1)

        # create blank input array and class label array
        X=np.zeros((len(windows),9000,1))
        Y=np.zeros((len(windows),1))

        # shuffle the samples
        shuffle(windows)

        # populate the arrays
        for w in range(len(windows)):
            X[w]=windows[w][0]
            Y[w]=windows[w][1]

        # class weights approximate data distribution
        # note models already have a bias towards avoiding false negatives due to class weights from initial training
        # (i.e. from a clinical viewpoint, they favour sensitivity over specificity)
        class_weight={0:1.,1:2.}

        # create validation data:
        windows=[]
        for filename in glob.iglob(file_path+'0\\*.npy'):
            patient=filename.split('\\')[-1].split('.')[0]
            if patient in excluded_list:
                x=np.load(filename)
                seconds=x.shape[0]//1000
                for h in range(seconds):
                    window=[x[h*1000:(h+1)*1000].reshape((9000,1)),0.]
                    windows.append(window)
                for h in range(seconds-1):
                    window=[x[(h*1000)+250:((h+1)*1000)+250].reshape((9000,1))*-1,0.]
                    windows.append(window)
                for h in range(seconds-1):
                    window=[x[(h*1000)+500:((h+1)*1000)+500].reshape((9000,1)),0.]
                    windows.append(window)
                for h in range(seconds-1):
                    window=[x[(h*1000)+750:((h+1)*1000)+750].reshape((9000,1))*-1,0.]
                    windows.append(window)
        print('\n')
        for filename in glob.iglob(file_path+'1\\*.npy'):
            patient=filename.split('\\')[-1].split('.')[0]
            if patient in excluded_list:
                x=np.load(filename)
                seconds=x.shape[0]//1000
                for h in range(seconds):
                    window=[x[h*1000:(h+1)*1000].reshape((9000,1)),1.]
                    windows.append(window)
                for h in range(seconds-1):
                    window=[x[(h*1000)+250:((h+1)*1000)+250].reshape((9000,1))*-1,1.]
                    windows.append(window)
                for h in range(seconds-1):
                    window=[x[(h*1000)+500:((h+1)*1000)+500].reshape((9000,1)),1.]
                    windows.append(window)
                for h in range(seconds-1):
                    window=[x[(h*1000)+750:((h+1)*1000)+750].reshape((9000,1))*-1,1.]
        print('\nLoaded',len(windows),'one-second validation windows for validaton set',i+1)

        X_val=np.zeros((len(windows),9000,1))
        Y_val=np.zeros((len(windows),1))

        shuffle(windows)

        for w in range(len(windows)):
            X_val[w]=windows[w][0]
            Y_val[w]=windows[w][1]

        # train the model

        # during feasibility study ('first-training.py'), weights were initiated from a model trained on an arrhythmia detection task
        # this is because it was felt that some of the previously-acquired knowledge regarding feature extraction from ECG signals would transfer
        # however, it was the first time the models had been trained for ischaemia detection
        # hence, the full models (i.e. all layers) were re-trained

        # now the aim is to fine-tune the models on extended data
        # hence, all but the final layer is frozen for the first training cycle

        # layers are incrementally unfrozen for subsequent cycles

        baseline_metrics=model.evaluate(x=X_val,y=Y_val,batch_size=128,verbose=1)
        print('Baseline model loss:',baseline_metrics[0],'Baseline model accuracy:',baseline_metrics[1])

        print('Data prepared. Training the model...')
        checkpointer=ModelCheckpoint('model_retrained_'+str(i)+'.h5',verbose=1,save_best_only=True)
        model.fit(x=X,y=Y,
                  epochs=3,
                  verbose=1,
                  batch_size=128,
                  callbacks=[checkpointer],
                  validation_data=(X_val,Y_val)
                  )

        print('Loading model',i+1,'for retraining of final convolutional layer...')
        model=load_model('model_retrained_'+str(i)+'.h5')
        for l in model.layers:
            l.trainable=True
        for l in model.layers[:-6]:
            l.trainable=False
        sgd=SGD(lr=0.0001)
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
        print('Model loaded. Re-training...')
        model.fit(x=X,y=Y,
                  epochs=1,
                  verbose=1,
                  batch_size=128,
                  callbacks=[checkpointer],
                  validation_data=(X_val,Y_val),
                  class_weight=class_weight
                  )

        print('Loading model',i+1,'for retraining of final two convolutional layers...')
        model=load_model('model_retrained_'+str(i)+'.h5')
        for l in model.layers:
            l.trainable=True
        for l in model.layers[:-10]:
            l.trainable=False
        sgd=SGD(lr=0.00001)
        model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['acc'])
        print('Model loaded. Re-training...')
        model.fit(x=X,y=Y,
                  epochs=1,
                  verbose=1,
                  batch_size=128,
                  callbacks=[checkpointer],
                  validation_data=(X_val,Y_val),
                  class_weight=class_weight
                  )

    evaluate_models(file_path,5,'new_data')

if __name__=="__main__":
    file_path="Patients\\"
    save_data(file_path)
    retrain_old_models_new_data(file_path)