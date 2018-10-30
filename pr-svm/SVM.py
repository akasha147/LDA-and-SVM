import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import cvxopt.solvers
import cvxopt
from numpy import linalg
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, args):

        self.X_train = args['X_train']

        print(self.X_train)

        self.Y_train = np.squeeze(args['Y_train'])
        self.kernel = args['kernel']
        self.C = args['c_param']
        self.n = self.X_train.shape[0]
        self.n_dims = self.X_train.shape[1]
        self.Y_train = np.where(self.Y_train == 0, -1 * np.ones_like(self.Y_train), self.Y_train)

        self.lambdas = None
        self.bias = None
        self.sv = None
        self.sv_y = None
        self.w = None

    def linear_kernel(self,x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self,x, y, p=2):
        return (1 + np.dot(x, y)) ** p

    def gaussian_kernel(self,x, y, sigma=5.0):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

    def kernel_function(self, x, y,p=2,sigma=5.0):
        if self.kernel == 'linear_kernel':
            return np.dot(x, y)
        if self.kernel == 'polynomial_kernel':
            return (1 + np.dot(x, y)) ** p
        if self.kernel == 'gaussian_kernel':
            return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

    def fit(self):
        # Gram matrix
        K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i,j] = self.kernel_function(self.X_train[i], self.X_train[j])

        P = cvxopt.matrix(np.outer(self.Y_train, self.Y_train) * K)
        q = cvxopt.matrix(np.ones(self.n) * -1)
        tmp = self.Y_train.astype(np.double)
        A = cvxopt.matrix(tmp, (1,self.n))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(self.n) * -1))
            h = cvxopt.matrix(np.zeros(self.n))
        else:
            tmp1 = np.diag(np.ones(self.n) * -1)
            tmp2 = np.identity(self.n)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(self.n)
            tmp2 = np.ones(self.n) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.lambdas = a[sv]
        self.sv = self.X_train[sv]
        self.sv_y = self.Y_train[sv]

        # Intercept
        self.bias = 0
        for n in range(len(self.lambdas)):
            self.bias += self.sv_y[n]
            self.bias = np.sum(self.lambdas* self.sv_y * K[ind[n],sv])
        self.bias /= len(self.lambdas)

        # Weight vector
        if self.kernel == 'linear_kernel':
            self.w = np.zeros(self.n_dims)
            for n in range(len(self.lambdas)):
                self.w += self.lambdas[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.bias
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.lambdas, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel_function(X[i], sv)
                y_predict[i] = s
            return y_predict + self.bias

    def predict(self, X):
        return np.sign(self.project(X))
