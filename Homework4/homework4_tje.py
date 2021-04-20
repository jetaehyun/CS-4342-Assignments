from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm

class SVM4342 ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit (self, X, y):

        # bias
        onesRow = np.ones((X.shape[0], 1))
        X = np.hstack((X, onesRow))
    
        G = (-y*(X.T)).T
        P = np.identity(X.shape[1])
        q = np.zeros(X.shape[1])
        h = -1 * np.ones(X.shape[0])

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        # To avoid any annoying errors due to broadcasting issues, I recommend
        # that you flatten() the w you retrieve from the solution vector so that
        # it becomes a 1-D np.array.
        
        alphas = np.array(sol['x']).flatten()

        self.w = alphas[:-1]
        self.b = alphas[-1]
        # print(self.w)
        # print(self.b)
        # print("OUT")

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, x):
        # print(x.shape)
        # print(self.w.shape)
        # print(self.b.shape)
        labels = x.dot(self.w) + self.b
        labels[labels > 0] = 1
        labels[labels < 0] = -1
        labels[labels == 0] = np.random.choice([-1, 1])

        return labels

def test1 ():
    # Set up toy problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])

    # Train your model
    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print(svm4342.w, svm4342.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm4342 = SVM4342()
    svm4342.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_ - svm4342.w) + np.abs(svm.intercept_ - svm4342.b)
    print(diff)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")
    else:
        print("FAILED")

if __name__ == "__main__": 
    test1()
    for seed in range(5):
        test2(seed)
