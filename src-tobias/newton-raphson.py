import numpy as np
from sklearn import linear_model
from sklearn import datasets
from scipy.special import expit
from scipy.special import xlogy


def add_intercept_column(X):
    return np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))


class logistic_regression:
    def __init__(self, X, y):
        self.X = add_intercept_column(X)
        self.y = y

    def p(self, X, beta):
        # print(f'exp(X @ beta):{np.exp(X @ beta)}')
        return 1 / (1 + np.exp(-X @ beta))

    def gradient_hessian(self, beta):
        P = self.p(self.X, beta)
        # print(f'P: {P}')
        W = np.diag(P * (1 - P))
        return self.X.T @ (y - P), -self.X.T @ W @ self.X

    def log_likelihood(self, beta):
        # print(self.p(self.X, beta))
        Xbeta = self.X @ beta
        return np.sum(self.y * Xbeta) - np.sum(np.log(1 + np.exp(Xbeta)))

    def newton_raphson(self, start, iterations=100, threshold=1e-6, verbose=False):
        beta = start
        if verbose:
            print(f'Initial beta: {beta}')
        for i in range(iterations):
            beta_old = beta
            gradient, hessian = self.gradient_hessian(beta_old)
            beta = beta_old - np.linalg.pinv(hessian) @ gradient
            if verbose:
                print(f'Iteration no: {i}')
                # print(gradient.shape, hessian.shape)
                # print(f'gradient: {gradient}')
                # print(f'hessian: {hessian}')
                print(f'New beta: {beta}')
                print(f'Log likelihood: {self.log_likelihood(beta)}')
                print(f'L2 change {np.linalg.norm(beta - beta_old)}')
            if np.linalg.norm(beta - beta_old) < threshold or beta[0] != beta[0]:
                break
        return beta


def newton_raphson(start, funcJacobian, iterations=1000, threshold=1e-6, verbose=False):
    beta = start
    # if verbose:
    #     print(f'beta: {beta}')
    for i in range(iterations):
        if verbose:
            print(i)
        beta_old = beta
        func, jacobian = funcJacobian(beta_old)
        # print(f'Jacobian: {jacobian}')
        # print(f'func: {func}')
        beta = beta_old - np.linalg.pinv(jacobian) @ func
        if verbose:
            # print(func.shape, jacobian.shape)
            # print(f'func: {func}')
            # print(f'jacobian: {jacobian}')
            print(f'new beta: {beta}')
            print(f'L2 change {np.linalg.norm(beta - beta_old)}')
        if np.linalg.norm(beta - beta_old) < threshold or beta[0] != beta[0]:
            break
    return beta


def funcJacobianTest(beta):
    func = np.array([beta[0]**2 - 2 * beta[0] + beta[1]**2 - beta[2] + 1,
                     beta[0] * beta[1]**2 - beta[0] - 3 * beta[1]
                     + beta[1] * beta[2] + 2,
                     beta[0] * beta[2]**2 - 3 * beta[2] + beta[1] * beta[2]**2
                     + beta[0] * beta[1]])
    jacobian = np.array([[2 * beta[0] - 2, 2 * beta[1], -1],
                         [beta[1]**2 - 1, 2 * beta[0] *
                             beta[1] - 3 + beta[2], beta[1]],
                         [beta[2]**2 + beta[1], beta[2]**2 + beta[0],
                          2 * beta[0] * beta[2] - 3 + 2 * beta[1] * beta[2]]])
    return func, jacobian


def cost(X, y, beta):
    m = X.shape[0]
    p = expit(X @ beta)
    return -np.mean(xlogy(y, p) + xlogy(1 - y, 1 - p))


# [1. 1. 1]
# print(newton_raphson(np.array([1, 2, 3]), funcJacobianTest, verbose=True))
# # [1.09894258 0.36761668 0.14493166]
# print(newton_raphson(np.array([0, 0, 0]), funcJacobianTest, verbose=True))

iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
# print(X)
# print(y)

logreg = logistic_regression(X, y)
beta = logreg.newton_raphson(np.zeros(3), 30, 1e-6, True)
print(f'beta: {beta}')

model = linear_model.LogisticRegression(C=1)
model.fit(X, y)
print(model.intercept_, model.coef_)
# Z = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
# print(cost(Z, y,
#            np.array([-3089.80863304, 996.19935741, -702.15535228]), 0))
# print(logreg.log_likelihood(np.array([-0.80059233, 2.48966566, -3.99890697])))


"""
X = np.array([[-34.01090879, 14.91883927,  23.3297436,  23.22881119, 52.52092729],
              [-28.90524496, 55.24885135,  29.57192817,  23.76951731, 22.718818],
              [-25.62986139, 9.56156831,  27.45077484,  47.36628958, 16.61804482],
              [8.41938725, 3.27601985,   3.53473298,  10.45798447, -12.72287755],
              [28.18573399, -16.19926955,  -0.22660912,  19.1534225, 50.74726269],
              [27.70351874, 57.27223336,  25.61636444,   7.01415298, 24.76966727],
              [27.71871261, 6.53224506,  11.70289757,   5.71284346, -10.44898657],
              [12.16841666, 4.60228889,  27.86493384,  24.28402065, -20.87553236],
              [-16.76739464, -9.87944463, -22.2200689, -14.59448948, -16.67160281],
              [15.15828199, -31.94945811,  55.73493626,   6.3957773, 11.55212357]]) / 50

y = np.array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0])

logreg = logistic_regression(X, y)
beta = logreg.newton_raphson(np.array(
    [-0.003489, -0.55942109, 0.87608081, -0.11086754, -0.30659983, -0.02973685]), 1000, 1e-6)
print(f'beta: {beta}')

logreg = linear_model.LogisticRegression(random_state=1, verbose=1, max_iter=1E3, tol=1E-5,
                                         solver='liblinear')

# fit training data
logreg.fit(X, y)

print(logreg.coef_)
print(logreg.intercept_)
"""
