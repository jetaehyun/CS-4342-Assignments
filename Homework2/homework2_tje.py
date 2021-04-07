import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    d = d + 1
    reg_x = np.ones((d, x.shape[0]))

    for i in range(d):
        reg_x[i] = np.power(x, i)

    w = np.linalg.solve(np.dot(reg_x, np.transpose(reg_x)), np.dot(reg_x, y))
    return w

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    faces = np.transpose(faces.reshape(-1, 48, 48))
    newShape = np.reshape(faces, (faces.shape[0]**2, faces.shape[2]))
    onesRow = np.ones(newShape.shape[1])
    newShape = np.vstack([newShape, onesRow])

    return newShape

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    yhat = np.dot(np.transpose(Xtilde), w)
    fmse = np.sum(np.square(yhat - y)) / (2*len(y))
    
    return fmse

def error(w, Xtilde, y):
    yhat = np.dot(np.transpose(Xtilde), w)
    big_errors = []

    diff = abs(y - yhat)
    diff = np.argsort(diff)[::-1][:5]

    for i in range(5):

        idx = diff[i]
        err_info = (yhat[idx], y[idx], idx)
        
        big_errors.append(err_info)

    print(f'5 biggest errors(yhat, ground truth y, and img #):\n{big_errors}\n')
    return big_errors


def RMSE(w, Xtilde, y):
    accuracy = 0.0
    labels = len(w)
    yhat = np.dot(np.transpose(Xtilde), w)

    for i in range(labels):
        accuracy += (yhat[i] - y[i])**2

    accuracy = (accuracy/labels)**(1/2)
    print(f'RMSE(in years): {accuracy}')

    return (accuracy / labels)**(1/2)


# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    yhat = np.dot(np.transpose(Xtilde), w)
    fmse = np.sum(np.square(yhat - y))

    penalty = np.dot(alpha / (2 * len(y)), np.dot(np.transpose(w), w))
    
    return (fmse + penalty) / (2*len(y))

    

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):

    xByx = np.dot(Xtilde, np.transpose(Xtilde))
    xByy = np.dot(Xtilde, y)

    w = np.linalg.solve(xByx, xByy)
    return w

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    w = 0.01 * np.random.randn(Xtilde.shape[0])
    for i in range(T):

        gradf = np.dot(Xtilde, np.dot(np.transpose(Xtilde), w) - y) / len(y)

        gradf[:-1] = gradf[:-1] + (alpha / len(y) * w[:-1])
        w -= EPSILON * gradf 
    return w

def displayImages(fileName, images):
    faces = np.load(fileName)
    faces = faces.reshape(-1, 48, 48)

    for imageNumber in images:
        im = faces[imageNumber,:,:]
        plt.figure(1); plt.clf()
        plt.imshow(im, cmap='gray')
        plt.show()



if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    # x = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
    # y = np.array([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])
    # wPoly = trainPolynomialRegressor(x, y, 3)
    imgList = [884, 1640, 830, 581, 939]
    displayImages("age_regression_Xte.npy", imgList)

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    w_list = [w1, w2, w3]

    RMSE(w3, Xtilde_te, yte)
    error(w3, Xtilde_te, yte)
    
    for i in w_list:
        img = plt.imshow(np.transpose(np.reshape(i[:-1], (48,48))))
        plt.show()

    
    print(f'Method 1 Training:\t{fMSE(w1, Xtilde_tr, ytr)}')
    print(f'Method 1 Testing:\t{fMSE(w1, Xtilde_te, yte)}\n')
    
    print(f'Method 2 Training:\t{gradfMSE(w2, Xtilde_tr, ytr)}')
    print(f'Method 2 Testing:\t{gradfMSE(w2, Xtilde_te, yte)}\n')

    print(f'Method 3 Training:\t{gradfMSE(w3, Xtilde_tr, ytr)}')
    print(f'Method 3 Testing:\t{gradfMSE(w3, Xtilde_te, yte)}\n')

    # Report fMSE cost using each of the three learned weight vectors
    # ...
