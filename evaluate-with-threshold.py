import numpy as np
from load_samples import load_samples

def evaluate_model_on_examples(threshold):
    samples=load_samples()

    y_pred=np.load('y_pred.npy')
    y_true=np.load('y_true.npy')
    y_counter=0

    csv='ECG,Prediction,Truth,Correct'

    for i in range(len(samples)):
        s=samples[i]
        # predict the baseline ECG
        y_non_ischaemic=int((y_pred[i*2]>threshold)*1)
        csv+='\n'+str(s[0]).zfill(3)+'a,'+str(y_non_ischaemic)+',0,'
        if y_non_ischaemic==0:
            csv+='1'
        else:
            csv+='0'
        # predict the balloon inflation ECG
        y_ischaemic=int((y_pred[(i*2)+1]>threshold)*1)
        csv+='\n'+str(s[0]).zfill(3)+s[4]+','+str(y_ischaemic)+',1,'
        if y_ischaemic==1:
            csv+='1'
        else:
            csv+='0'

    # write results to file
    f=open('db_results.csv','w')
    f.write(csv)
    f.close

def find_optimal_threshold():
    f=open('optimal_threshold_search.csv','r')
    thresh_csv=f.read()
    f.close()
    thresh_csv=thresh_csv.split('\n')[1:]
    thresh=0
    max_f1=0
    for t in thresh_csv:
        temp_thresh_list=[]
        results_list=t.split(',')
        if float(results_list[3]) > max_f1:
            max_f1=float(results_list[3])
            thresh=float(results_list[5])
    print('Max F1:',max_f1,'Threshold:',thresh)
    return thresh

if __name__=="__main__":
    evaluate_model_on_examples(find_optimal_threshold())