
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time

#directory='/Users/seoyoung/PycharmProjects/Dataminng/Assignment2_v3/Q1_dataset'
Test =pd.read_csv("/Users/seoyoung/PycharmProjects/Dataminng/Assignment2_v3/Q1_dataset/letter_test.csv")
Train=pd.read_csv("/Users/seoyoung/PycharmProjects/Dataminng/Assignment2_v3/Q1_dataset/letter_train.csv")


num_feature=16 # number of feature

# Splitting into train sets and test sets
x_train=Train.iloc[:,1:]
y_train=Train.iloc[:,0]

x_test=Test.iloc[:,1:]
y_test=Test.iloc[:,0]

time_entropy=[] #training time
time_gini=[]

accuracy_entropy = []#accuracy
accuracy_gini = []

f1_entropy=[]#f1-score
f1_gini =[]

pre_entropy=[]#precision
pre_gini=[]

re_entropy=[]#recall
re_gini=[]

for depth in range(5, 30, 5):
    # print(depth)

    # train model
    classifier_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    classifier_gini = DecisionTreeClassifier(criterion='gini', max_depth=depth)

    # training time
    t0 = time.time()
    classifier_entropy.fit(x_train, y_train)
    time_entropy.append(round(time.time() - t0, 3))

    t1 = time.time()
    classifier_gini.fit(x_train, y_train)
    time_gini.append(round(time.time() - t1, 3))

    # predict
    predicted_gini = classifier_gini.predict(x_test)
    predicted_entropy = classifier_entropy.predict(x_test)

    # acurracy score: the mean of accruacy on the given data : accuracy on test data
    accuracy_entropy.append(classifier_entropy.score(x_test, y_test))
    accuracy_gini.append(classifier_gini.score(x_test, y_test))

    # f1-score
    f1_entropy.append(f1_score(y_test, predicted_entropy, average='weighted'))
    f1_gini.append(f1_score(y_test, predicted_gini, average='weighted'))

    # recall
    re_entropy.append(recall_score(y_test, predicted_entropy, average='weighted'))
    re_gini.append(recall_score(y_test, predicted_gini, average='weighted'))

    # precision
    pre_entropy.append(precision_score(y_test, predicted_entropy, average="weighted"))
    pre_gini.append(precision_score(y_test, predicted_gini, average="weighted"))

print("========================complete train the trees=========================")



#dataframe
df_entropy =pd.DataFrame([time_entropy,accuracy_entropy,f1_entropy,re_entropy,pre_entropy],
                         columns=["5","10","15","20","25"],index=['time', 'accuracy',"f1_score","recall","precision"])

df_gini =pd.DataFrame([time_gini,accuracy_gini,f1_gini,re_gini,pre_gini],
                         columns=["5","10","15","20","25"],index=['time', 'accuracy',"f1_score","recall","precision"])

print("entropy result")
print(df_entropy)

print("gini result")
print(df_gini)

