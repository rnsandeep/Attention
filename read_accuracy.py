import pickle
import sys, os
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
mypath = sys.argv[1]
onlyfiles = [os.path.join(mypath, f )for f in listdir(mypath) if isfile(join(mypath, f))]
max_acc = 0
mean_acc = 0
def transpose(labels):
    labels = np.array(labels)
    labels[np.where(labels==0)] = 2
    labels[np.where(labels==1)] = 0
    labels[np.where(labels==2)] = 1

    return list(labels)

def calculatePrecisionRecallAccuracy(labels, outputs):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for label, output in zip(labels, outputs):
        if label==output and label ==0:
            tn = tn+1
        elif label==output and label!=0:
            tp = tp+1
        elif label!=output and label==0:
            fp = fp+1
        else:
            fn = fn +1
    if tp+fp ==0:
       precision = 0  
    else:   
       precision = tp*1.0/(tp+fp)
    if tp+fn ==0:   
       precision= 0
    else:   
       recall = tp*1.0/(tp+fn)
    if tp+fp+fn+tn ==0:
       accuracy = 0
    else:   
       accuracy = (tp+tn)*1.0/(tp+fp+fn+tn)
#    print("tp:", tp, "fp:", fp, "fn:", fn)
    return  precision, recall, accuracy

onlyfiles.sort(key=lambda x: (int(os.path.basename(x).split('_')[0]), x) )

for f in onlyfiles:
    basename = os.path.basename(f)
    splits = basename.split('_')
    epoch = splits[0]
    scale = splits[2]
    resize = splits[3]
    accuracy, labels, outputs = pickle.load(open(f,'rb'))
    labels = np.array(labels)
    outputs = np.array(outputs)

    classes = ['no-pigm',  'z1-perioral-pigm',  'z2-foreheadpigm',  'z3-cheekpigm',  'z4-nosepigm']

    for i in range(0, np.max(labels)+1):
        p = np.where(labels==outputs)
        tp = len(np.where(labels[p]==i)[0])
        total = len(np.where(labels == i)[0])
        total2 = len(np.where(outputs == i)[0])
#        print(tp, total, tp, total2, "recall:", tp*1.0/total, "precision:", tp*1.0/total2, classes[i])
        
#    print(classes)
#    a = classification_report( labels, outputs)
#    print("precision and recall:")
#    print(a)
    
    precision, recall, accuracy = calculatePrecisionRecallAccuracy(labels, outputs)
#    print(classification_report(labels, outputs))
    matrix = confusion_matrix(labels, outputs)
#    print(matrix)
    if max_acc < accuracy:
      max_acc = accuracy
    mean_acc += accuracy
    print("precision:%.3f"%precision, "recall:%.3f"%recall,"accuracy:%.3f"%accuracy, "epoch:", epoch, "scale;", scale, "resize:", resize)

print("maximum accuracy:", max_acc)
print("Mean_accuracy:", mean_acc/len(onlyfiles))
