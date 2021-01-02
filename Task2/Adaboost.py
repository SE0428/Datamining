import pandas as pd
import numpy as np
import random
import math

class Adaboost():

    def __init__(self):
        # initiate threshold 0
        self.threshold = []
        self.w = np.full(10, (1 / 10))  # len(y) #inital weight
        self.round = 3  # total # of round
        self.importance = []
        self.newSample = None

    def setdata(self,x,y):
        self.x = x
        self.y = y

    def sign(self, x):
        if x > 0:
            return 1
        else:
            return -1

    def fit(self):
        # by changing threshold of classifier
        Expected = []  # for expected label

        for i in range(self.round):
            """self.x = x  # get resampling data
            self.y = y"""

            x = self.x
            y = self.y

            print(i + 1, "round dataset")
            print(x)
            print(y)

            # temp threshold
            threshold = round(random.uniform(0, 1) * 10) + 0.5  # choose random threshold for next
            self.threshold.append(threshold)

            # print(self.threshold)

            print("threshold", threshold)
            print("weight", self.w)

            # print("weight",self.w) #BEFORE noramlize

            for j in range(len(y)):

                if (x[j] > threshold):
                    # expectedValue=1
                    Expected.append(1)

                else:
                    # expectedValue=-1
                    Expected.append(-1)

            # here we update weighted for each round
            if i <= self.round - 1:  # not to update at the last stage
                self.w = self.updateWeight(Expected, y)
                self.resampling()

            # clear temp for next round
            Expected = []

        self.finalClassifier(x, y)

    def updateWeight(self, Expected, y):
        w = self.w  # len(y)
        # temp : Expected Value from classifer
        # y: Trule label

        # calculate total error rate with weight
        print("Expected", Expected)
        print("Actal Label", y)

        error = 0  # initiate error
        newWeight = []

        for i in range(len(y)):
            if y[i] != Expected[i]:
                error += 1 * self.w[i]

        print("error for this round", error)
        print("======================================================================================")

        # calculate importance of classifier
        importance = 0.5 * math.log((1.0 + 1e-10 - error) / (error + 1e-10))
        self.importance.append(importance)
        # in case, dealing with error = 0

        # calculate updated weight with nomalization (sum of weights = 1)
        for i in range(len(y)):
            if y[i] != Expected[i]:  # error -- > weighet increase
                newWeight.append(w[i] * np.exp(importance))

            else:  # correct --> weight decrease
                newWeight.append(w[i] * np.exp(-importance))

        newWeight = newWeight / sum(newWeight)
        self.w = newWeight

        return self.w

    def resampling(self):
        newX = [0 for _ in range(len(y))]
        newY = [0 for _ in range(len(y))]
        """
        Each training the new sample is generated from training set, with diffrent sampling probablity according to weights
        
        given datapoint as distribution, find closest point then take it as sample

        allow duplicate 
        """
        #sampling probability
        series = pd.Series(self.w)
        cumsum = series.cumsum()


        #print("cumsum",cumsum)

        idx=[]
        temp_distance=[]

        for j in range(len(y)):

            for i in range(len(y)):
                temp = random.uniform(0, 1)
                temp_distance = [abs(x - temp) for x in cumsum]

            #index for Sampled data point
            idx.append(temp_distance.index(min(temp_distance)))

        for i in range(len(y)):

            newX[i] = x[idx[i]]
            newY[i] = y[idx[i]]


        self.x = newX
        self.y = newY


    def finalClassifier(self, x, y):
        """
            You should also report the final expression of the strong classifier,
            such as C∗(x) = sign[α1C1(x)+ α2C2(x) + α3C3(x) + . . .], where Ci(x)
            is the base classifier and αi is the weight of base classifier.
            You are also required to describe each basic classifier in detail.

        """
        print("importace for each classifier", self.importance)

        Expected = [0 for _ in range(len(y))]
        temp = [0 for _ in range(len(y))]

        for i in range(self.round):
            for j in range(len(y)):
                # print(j)
                # [α1C1(x)+ α2C2(x) + α3C3(x) + . . .]
                if x[j] > self.threshold[i]:
                    # classify as 1 * importance
                    temp[j] += self.importance[i] * 1

                else:
                    temp[j] += self.importance[i] * (-1)

        for i in range(len(y)):
            Expected[i] = self.sign(temp[i])

        print("      Actual   Label", y)
        print("Final Expected Label", Expected)
        return None



x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]



random.seed(20) #to get the same result
clf=Adaboost()
clf.setdata(x,y)
clf.fit()
