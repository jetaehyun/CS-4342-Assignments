import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A,B)-C

def problem3 (A, B, C):
    return np.array(A)*np.array(B)+np.transpose(C)

def problem4 (x, S, y):
    return np.dot(np.transpose(x), np.dot(S, y))

def problem5 (A):
    return np.zeros((len(A),len(A[0])))

def problem6 (A):
    return np.ones((len(A)))

def problem7 (A, alpha):
    return A+(alpha*np.eye(len(A)))

def problem8 (A, i, j):
    return A[i][j]

def problem9 (A, i):
    return np.sum(A[i])

def problem10 (A, c, d):
    A = np.array(A)
    return np.mean([i for i in A[np.nonzero(A)] if i >= c and i <= d])

def problem11 (A, k):
    eVal,eVec = np.linalg.eig(A)
    ind = eVal.argsort()
    newMatrx = np.zeros(shape=(len(A), k))
    
    for i in range(k):
        newMatrx[:,i] = eVec[:,ind[len(A) - k + i]]

    return newMatrx

def problem12 (A, x):
    return np.linalg.solve(A, x)

def problem13 (A, x):
    return np.linalg.solve(np.transpose(A), np.transpose(x))


# b = [[8,0,1],[0,3,5],[4,1,3]]
# a = [[11,3],[7,12]]
# c = [[1,1,1],[1,1,1]]

# print(problem11(b, 1))
