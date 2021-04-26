import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

def phiPoly3 (X):
    return np.hstack((np.ones_like(X[:,[0]]), 3**.5 * X[:,[1]], 3**.5 * X[:,[1]]**2, X[:,[1]]**3, 3**.5 * X[:,[0]], 6**.5 * X[:,[0]] * X[:,[1]], 3**.5 * X[:,[0]] * X[:,[1]]**2, 3**.5 * X[:,[0]]**2, 3**.5 * X[:,[0]]**2 * X[:,[1]], X[:,[0]]**3))


def kerPoly3 (x, xprime):
    return (1 + x.T.dot(xprime))**3


def trans10Dim(u, v):
    feat = np.ones((10, 1))

    feat[0] = 1
    feat[1] = 3**.5 * v
    feat[2] = 3**.5 * v**2
    feat[3] = v**3
    feat[4] = 3**.5 * u
    feat[5] = 6**.5 * u * v
    feat[6] = 3**.5 * u * v**2
    feat[7] = 3**.5 * u**2
    feat[8] = 3**.5 * u**2 * v
    feat[9] = u**3


    return feat.T


def kernelGrid(X):
    K = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        x_it = X[i]
        
        for j in range(X.shape[0]):
            x_prime = X[j]

            k_it = kerPoly3(x_it, x_prime)

            K[i,j] = k_it



    return K

def kernelPredict(X, feat_pairs):
    feat = np.zeros((1, 100))
    
    for i in range(X.shape[0]):
        feat[0][i] = kerPoly3(feat_pairs, X[i])

    return feat


def showPredictions (title, svm, X, svm_type=0):  # feel free to add other parameters if desired

    x_neg, y_neg = [], []
    x_pos, y_pos = [], []
 
    x_grid = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    y_grid = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)
    length = len(x_grid)

    for i in range(length):
        for j in range(length):

            xp, yp = x_grid[i], y_grid[j]
            point = 0

            if svm_type == 0:
                point = np.array([[xp, yp]])
            elif svm_type == 1:
                point = trans10Dim(xp, yp)
            else:
                point = kernelPredict(X, np.array([xp, yp]))
            

            predi = svm.predict(point)


            if predi == -1:
                x_neg.append(xp)
                y_neg.append(yp)
            else:
                x_pos.append(xp)
                y_pos.append(yp)


    plt.scatter(x_neg, y_neg, label='negative')
    plt.scatter(x_pos, y_pos, label='positive')

    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend([ "Lung disease", "No lung disease" ], loc='upper right')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Load training data
    d = np.load("lung_toy.npy")
    X = d[:,0:2]  # features
    y = d[:,2]  # labels
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1])
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1])
    plt.show()


    # (a) Train linear SVM using sklearn
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    showPredictions("Linear", svmLinear, X)


    # (b) Poly-3 using explicit transformation phiPoly3
    Xtilde = phiPoly3(X)
    svmPoly3_expl = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmPoly3_expl.fit(Xtilde, y)
    showPredictions("phiPoly3", svmPoly3_expl, X, 1)
  

    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    K = kernelGrid(X)
    svmPoly3_ker = sklearn.svm.SVC(kernel='precomputed', C=0.01)
    svmPoly3_ker.fit(K, y)
    showPredictions("kerPoly3", svmPoly3_ker, X, 3)
  

    # (d) Poly-3 using sklearn's built-in polynomial kernel
    svmPoly3_sk = sklearn.svm.SVC(kernel='poly', C=0.01, gamma=1, coef0=1, degree=3)
    svmPoly3_sk.fit(X, y)
    showPredictions("Poly3_sk", svmPoly3_sk, X)
  

    # (e) RBF using sklearn's built-in polynomial kernel
     
    svmPoly3_rbf_1 = sklearn.svm.SVC(kernel='rbf', C=1, gamma=0.1)
    svmPoly3_rbf_1.fit(X, y)
    showPredictions("Poly3_rbf 0.1", svmPoly3_rbf_1, X)

    svmPoly3_rbf_2 = sklearn.svm.SVC(kernel='rbf', C=1, gamma=0.03)
    svmPoly3_rbf_2.fit(X, y)
    showPredictions("Poly3_rbf 0.03", svmPoly3_rbf_2, X)
