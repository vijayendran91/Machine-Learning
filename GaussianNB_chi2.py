import sklearn

__author__ = 'vijay'

from matplotlib import pyplot as plt
import time
from sklearn.naive_bayes import GaussianNB
import numpy as np
#from sklearn.feature_selection import VarianceThreshold
import sklearn.linear_model
start = time.time()
datafile = open("/home/vijay/PycharmProjects/ML-Final/traindata")
data = []
la = datafile.readline()
while (la != ''):
    a = la.split()
    l2 = []
    for i in range(0, len(a)):
        l2.append(int(a[i]))

    data.append(l2)
    la = datafile.readline()
print ("row length= ", len(data))
print ("Column length= ", len(data[0]))


data = np.array(data)
split10 = int(len(data) * .10)
data10 = data[0:split10]
data90 = data[split10:]
print("90% of traindata's row length", len(data90))
print("10% of the traindata's row length", len(data10))
datafile.close()
end = time.time()
print "Time Taken : ",end-start
start=time.time()



datafile = open("/home/vijay/PycharmProjects/ML-Final/trainlabels")
n_count = []
n_count.append(0)
n_count.append(0)
train_labels = []
l = datafile.readline()
while (l != ''):
    a = l.split()
    train_labels.append(int(a[0]))
    l = datafile.readline()
    n_count[int(a[0])] += 1



train_labels = np.array(train_labels)
label10 = train_labels[0:split10]
label90 = train_labels[split10:]
end=time.time()

clf= sklearn.linear_model.SGDClassifier()
clf.fit(data90,label90)





#print "Time Taken : ",end-start
#gnb = GaussianNB()
#gnb.fit(data90, label90)
#print data90.shape
#print label90.shape

res = []
for x in data10:
    res.append(clf.predict(x))

from sklearn.metrics import accuracy_score
print accuracy_score(res,label10)



#reduced_data=sklearn.feature_selection.chi2(data90,label90)
#print reduced_data
##http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score