# Datamining
2020Spring

# Task1 : Comparison of Classifiers

### Data Description

*  the letter dataset from Statlog
* The class number of the dataset is 26

### Classifiers

* Decision Tree
* KNN, Random Forest

### Result

Here the dataset we use is well balanced and for us and it is more important to focus on the true positive case, Otherwise, it would not mean anything if we classify the wrong which is letter that could misleading the meaning of word. So, recall score is useful for measuring model performance. The training time is not really different between each other’s. So, I would say decision tree with entropy works well than decision tree with Gini for letter classification.Random forest takes a lot more time than KNN even though the max depth was 5 which is very small considering the total number of datasets. However, its accuracy is already high, showing around 95% accuracy. So, given the dataset we have, we need to take different tactics. If we can handle the computational cost and training time, Random Forest will be the better solution. If not, KNN is good alternative for Random forest for this dataset.



# Task2 : Implementation of Adaboost

### Data Description

* 10 data and 2 labels

### Classifiers

* Adaboost

### Results


Adaboost is following those four steps:
1) Adaboost combines a lot of weak leaners to make classification, called stumps
2) Weight gets updated in each step, emphasizing the misclassified datapoint
3) Some stumps get more importance in classification than others based on their performance
4) In every step, resampling the sample form the original dataset, allowing the get the same
datapoint many times in new sampling dataset

The accuracy of classifier is getting higher and higher by adjusting weight, importance and resampling. 
The first-round error was 0.6. However, the final classifier gets 0.3 error rate, which is half than the first trial.

