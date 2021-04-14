import pandas
import numpy as np
import random as random
from homework3_tje import *

if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    sibSp = d.SibSp.to_numpy()

    # Train model using part of homework 3.
    # ...

    titanicTrain = np.vstack((sex, Pclass, sibSp)).T
    
    shuffler_tr = np.random.permutation(len(titanicTrain))
    titanicTrain = transImage(titanicTrain, shuffler_tr)
    y = one_hot_label(y, shuffler_tr)

    w = softmaxRegression(titanicTrain, y, titanicTrain, y, 0.1, 27)

    # Load testing data
    # ...

    d_te = pandas.read_csv("test.csv")
    sex_te = d_te.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass_te = d_te.Pclass.to_numpy()
    sibSp_te = d_te.SibSp.to_numpy()

    # Compute predictions on test set
    # ...
    
    titanicTest = np.vstack((sex_te, Pclass_te, sibSp_te)).T
    titanicTest = transImage(titanicTest, [i for i in range(titanicTest.shape[0])])
    yhat = softmax(titanicTest, w).T

    # Write CSV file of the format:
    # PassengerId, Survived
    # ..., ...

    predictions = [0] * yhat.shape[0]
    passengerList = yhat.shape[0]
    
    for i in range(passengerList):
        pred_pair = yhat[i]
        pred1, pred2 = pred_pair[0], pred_pair[1]
        
        predictions[i] = 1 if pred1 < pred2 else 0

    pd = pandas.read_csv("submission_.csv")
    pd.insert(1, "Survived", predictions)

    pd.to_csv("submission.csv", index=False)
