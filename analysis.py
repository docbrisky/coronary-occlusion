from sklearn import metrics as skmetrics
import numpy as np

f=open('db_results.csv','r')
results=f.read()
f.close()

results=results.split('\n')[1:]

estimates=[]
ground_truth=[]

for r in results:
    ground_truth.append(int(float(r.split(',')[2])))
    estimates.append(int(float(r.split(',')[1])))

tp=0
tn=0
fp=0
fn=0
correct_diagnoses=0

total_results=[]
# key for each results array: [tp,fp,fn,tn]
for i in range(5):
    total_results.append([0]*4)

model_number=0

for i in range(len(estimates)):
    if (i%30==0 and i!=0 and model_number<4):
        model_number+=1
    if estimates[i]==ground_truth[i]:
        correct_diagnoses+=1
        if (estimates[i]==0):
            tn+=1
            total_results[model_number][3]+=1
        else:
            tp+=1
            total_results[model_number][0]+=1
    else:
        if (estimates[i]==0):
            fp+=1
            total_results[model_number][2]+=1
        else:
            fn+=1
            total_results[model_number][1]+=1

summary=""
total_metrics=[]
# key for each metrics array: [sensitivity, specificity, PPV, accuracy, F1 score]
for i in range(5):
    metrics=[]
    metrics.append(total_results[i][0]/(total_results[i][0]+total_results[i][2]))
    metrics.append(total_results[i][3]/(total_results[i][3]+total_results[i][1]))
    metrics.append(total_results[i][0]/(total_results[i][0]+total_results[i][1]))
    metrics.append((total_results[i][0]+total_results[i][3])/(total_results[i][0]+total_results[i][1]+total_results[i][2]+total_results[i][3]))
    metrics.append(2*((metrics[0]*metrics[2])/(metrics[0]+metrics[2])))
    total_metrics.append(metrics)
    summary+='Model '+str(i)+':\n'
    summary+='Sensitivity: '+str(metrics[0])+' Specificity: '+str(metrics[1])+' PPV: '+str(metrics[2])+' Accuracy: '+str(metrics[3])+' F1: '+str(metrics[4])+'\n\n'

sens=tp/(tp+fn)
spec=tn/(tn+fp)
ppv=tp/(tp+fp)
acc=correct_diagnoses/152
f1=2*((sens*ppv)/(sens+ppv))

summary+='Average for 5-fold cross validation:\n'
summary+='Sensitivity: '+str(sens)+' Specificity: '+str(spec)+' PPV: '+str(ppv)+' Accuracy: '+str(acc)+' F1: ' + str(f1)

pred=np.load('y_pred.npy')
y=np.load('y_true.npy')

y=y.reshape(y.shape[0])
pred=pred.reshape(y.shape[0])

fpr, tpr, thresholds = skmetrics.roc_curve(y, pred, pos_label=1.)
auc=skmetrics.auc(fpr, tpr)

summary+='\n\nArea under the receiver operating characteristic curve: '+str(auc)

f=open('db_results.csv','r')
results=f.read()
f.close()

confusion_matrix=',Predicted:YES,Predicted:NO\nActual:YES,'+str(tp)+','+str(fn)+'\nActual:NO,'+str(fp)+','+str(tn)+'\n'

f=open('Confusion_matrix.csv','w')
f.write(confusion_matrix)
f.close()

f=open('Analysis_summary.txt','w')
f.write(summary)
f.close()

print(summary)