import numpy as np

if __name__=="__main__":
    y=np.load('y_pred.npy')
    print(y.shape)
    y_true=np.load('y_true.npy')
    Y_pred=np.zeros((y.shape[0],2))
    Y_pred[:,0]=y[:,0]
    Y_pred[:,1]=y_true[:,0]

    print('Shape',Y_pred)

    max_sens=[0,0,0,0,0,0]
    max_f1=[0,0,0,0,0,0]
    max_acc=[0,0,0,0,0,0]
    sens_min_90=[0,0,0,0,0,0]
    sens_min_90_switch=False

    for i in range(1,1000):
        threshold=i/1000
        tp=0.0001
        tn=0.0001
        fp=0.0001
        fn=0.0001
        acc=0
        for h in range(Y_pred.shape[0]):
            y_theta=Y_pred[h,0]>threshold*1.
            y_true=Y_pred[h,1]
            if y_true==0:
                if y_theta==0:
                    tn+=1
                    acc+=1
                else:
                    fn+=1
            if y_true==1:
                if y_theta==1:
                    tp+=1
                    acc+=1
                else:
                    fp+=1
        sens=tp/(tp+fn)
        prec=tp/(tp+fp)
        spec=tn/(tn+fp)
        f1=(2*sens*prec)/(sens+prec)
        acc=acc/154
        if (sens>max_sens[0]):
            max_sens=[sens,spec,f1,prec,threshold,acc]
        if (f1>max_f1[2]):
            max_f1=[sens,spec,f1,prec,threshold,acc]
        if (acc>max_acc[-1]):
            max_acc=[sens,spec,f1,prec,threshold,acc]
        if (sens>0.9 and sens_min_90_switch==False):
            sens_min_90=[sens,spec,f1,prec,threshold,acc]
            sens_min_90_switch=True

    csv=',Sensitivity,Specificity,F1 score,PPV,Threshold,Accuracy'
    csv+='\nMaximum Sensitivity:'
    for s in max_sens:
        csv+=','+str(s)
    csv+='\nMaximum F1:'
    for s in max_f1:
        csv+=','+str(s)
    csv+='\nMaximum accuracy:'
    for s in max_acc:
        csv+=','+str(s)
    csv+='\nSensitivity 90%:'
    for s in sens_min_90:
        csv+=','+str(s)

    f=open('optimal_threshold_search.csv','w')
    f.write(csv)
    f.close()
    print(csv)
