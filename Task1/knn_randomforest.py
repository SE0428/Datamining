import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

# import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# directory='/Users/seoyoung/PycharmProjects/Dataminng/Assignment2_v3/Q1_dataset'
Test = pd.read_csv("/Users/seoyoung/PycharmProjects/Dataminng/Assignment2_v3/Q1_dataset/letter_test.csv")
Train = pd.read_csv("/Users/seoyoung/PycharmProjects/Dataminng/Assignment2_v3/Q1_dataset/letter_train.csv")

x_train = Train.iloc[:, 1:]
y_train = Train.iloc[:, 0]

x_test = Test.iloc[:, 1:]
y_test = Test.iloc[:, 0]


# Setup arrays to store training and test accuracies
neighbors = np.arange(20, 40, 5)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

time_knn = []
time_RF = []

accuracy_knn = []  # accuracy
accuracy_RF = []

f1_knn = []  # f1-score
f1_RF = []

pre_knn = []  # precision
pre_RF = []

re_knn = []  # recall
re_RF = []

for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model
    t3 = time.time()
    knn.fit(x_train, y_train)
    time_knn.append(round(time.time() - t3, 3))  # time

    predicted_knn = knn.predict(x_test)

    # acurracy score: the mean of accruacy on the given data : accuracy on test data
    accuracy_knn.append(knn.score(x_test, y_test))

    # f1-score
    f1_knn.append(f1_score(y_test, predicted_knn, average='weighted'))

    # recall
    re_knn.append(recall_score(y_test, predicted_knn, average='weighted'))

    # precision
    pre_knn.append(precision_score(y_test, predicted_knn, average="weighted"))

print("========================complete train KNN=========================")

df_knn = pd.DataFrame([time_knn, accuracy_knn, f1_knn, re_knn, pre_knn],
                      columns=["20", "25", "30", "35"],
                      index=['time', 'accuracy', "f1_score", "recall", "precision"])

print("knn result")
print(df_knn)

max_feature= np.arange(4, 16, 3)
for i, k in enumerate(max_feature):
    #print(k)
    # Setup a knn classifier with RF

    RF= RandomForestClassifier(n_estimators=100,oob_score=True,max_features=k)

    # Fit the model
    t4 = time.time()
    RF.fit(x_train, y_train)
    time_RF.append(round(time.time() - t4, 3))  # time

    predicted_RF = RF.predict(x_test)

    # acurracy score: the mean of accruacy on the given data : accuracy on test data
    accuracy_RF.append(RF.score(x_test, y_test))

    # f1-score
    f1_RF.append(f1_score(y_test, predicted_RF, average='weighted'))

    # recall
    re_RF.append(recall_score(y_test, predicted_RF, average='weighted'))

    # precision
    pre_RF.append(precision_score(y_test, predicted_RF, average="weighted"))

print("========================complete train RF=========================")

df_RF = pd.DataFrame([time_RF, accuracy_RF, f1_RF, re_RF, pre_RF],
                      columns=["4", "7", "10", "13"],
                      index=['time', 'accuracy', "f1_score", "recall", "precision"])

print("RF result")
print(df_RF)
