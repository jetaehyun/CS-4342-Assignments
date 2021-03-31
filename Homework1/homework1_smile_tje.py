import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# y - ground truth labels of a set
# yhat - vector of guesses
def fPC (y, yhat):
    return np.mean(y==yhat)

# predictors - set of predictors (r1, c1, r2, c2)
# X - set of images
# y - ground truth labels of X
def measureAccuracyOfPredictors (predictors, X, y):
    yhatSum = []

    for pred in predictors:
        r1, c1 = pred[0], pred[1]
        r2, c2 = pred[2], pred[3]

        img = X[:,r1,c1] - X[:,r2,c2]

        img[img > 0] = 1
        img[img < 0] = 0
        
        yhatSum.append(img)
    
    yhatSum = np.sum(yhatSum, axis=0)
    yhat = np.divide(yhatSum, len(predictors))
    yhat[yhat > 0.5] = 1
    yhat[yhat <= 0.5] = 0

    return fPC(y, yhat)

def obtainPredictors(trainingFaces, trainingLabels, numberOfSamples):
    features = []

    trainingFaces = trainingFaces[:numberOfSamples]
    trainingLabels = trainingLabels[:numberOfSamples]

    for j in range(5):
        accuracy = -1
        best_predictor = None

        for r1 in range(24):
            for c1 in range(24):
                for r2 in range(24):
                    for c2 in range(24):

                        featuresCpy = features.copy()
                        featuresCpy.append((r1,c1,r2,c2))
                        predictor = (r1,c1,r2,c2)
                        
                        accuracyChk = measureAccuracyOfPredictors(featuresCpy, trainingFaces, trainingLabels)

                        if accuracyChk > accuracy and predictor not in features:
                            accuracy = accuracyChk
                            best_predictor = (r1,c1,r2,c2)

        features.append(best_predictor)

    trainingAccuracy = measureAccuracyOfPredictors(features, trainingFaces, trainingLabels)
    print(f'features: {features}')
    print(f'training accuracy: {trainingAccuracy*100} %')

    return features

def visualizeImage(testingFaces, features, imageNumber):

    patchList = []
    if len(features) == 0:
        return 

    fig,ax = plt.subplots(1)
    for i in features:
        im = testingFaces[imageNumber,:,:]
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        
        c1, r1, c2, r2 = i[0], i[1], i[2], i[3]
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        patchList.append(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        # Display the merged result
        patchList.append(rect)
 

    for i in patchList:
        ax.add_patch(i)

    plt.show()


# features: [(20, 17, 17, 7), (13, 4, 11, 14), (21, 8, 16, 8), (12, 5, 16, 17), (10, 6, 12, 6)]
# training accuracy: 82.75 %
# testing accuracy: 71.11597374179432 %

# features: [(20, 7, 17, 7), (13, 5, 11, 13), (18, 12, 16, 17), (12, 19, 10, 14), (19, 9, 13, 17)]
# training accuracy: 80.25 %
# testing accuracy: 74.07002188183807 %

# features: [(20, 7, 17, 7), (13, 5, 11, 13), (20, 17, 16, 17), (12, 19, 12, 13), (10, 7, 14, 7)]
# training accuracy: 79.83333333333333 %
# testing accuracy: 74.945295404814 %

# features: [(20, 7, 17, 7), (13, 6, 16, 17), (18, 12, 16, 7), (13, 5, 0, 19), (19, 12, 15, 17)]
# training accuracy: 78.875 %
# testing accuracy: 76.36761487964989 %

# features: [(20, 7, 17, 7), (12, 5, 10, 13), (20, 17, 16, 17), (11, 19, 12, 12), (19, 11, 14, 7)]
# training accuracy: 78.25 %
# testing accuracy: 76.42231947483589 %
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    show = True
    
    samples = [400, 800, 1200, 1600, 2000]
    for i in samples:
        features = obtainPredictors(trainingFaces, trainingLabels, i)
        accuracy = measureAccuracyOfPredictors(features, testingFaces, testingLabels)
        print(f'testing accuracy: {accuracy*100} %\n')


    features = [(20, 7, 17, 7), (12, 5, 10, 13), (20, 17, 16, 17), (11, 19, 12, 12), (19, 11, 14, 7)]
    if show:
        visualizeImage(testingFaces, features, 10)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
