# regressions.kernel_pls

"""A package which implements kernel PLS."""

import random

from . import *


class Kernel_PLS:

    """Regression using kernel PLS."""

    def __init__(self, X, Y, g, X_kernel,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 iteration_convergence=DEFAULT_EPSILON,
                 ignore_failures=True):

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same '
                                 'number of rows (data samples)')

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]

        self.X_training_set = X
        self.X_kernel = X_kernel

        K = np.empty((self.data_samples, self.data_samples))
        for i in range(0, self.data_samples):
            for j in range(0, self.data_samples):
                K[i, j] = X_kernel(X[i, :], X[j, :])

        centralizer = (np.identity(self.data_samples)) - \
            1.0 / self.data_samples * \
            np.ones((self.data_samples, self.data_samples))
        K = centralizer @ K @ centralizer
        self.K = K

        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset  # Yc is the centred version of Y

        T = np.empty((self.data_samples, g))
        Q = np.empty((self.Y_variables, g))
        U = np.empty((self.data_samples, g))
        P = np.empty((self.data_samples, g))

        self.components = 0
        K_j = K
        Y_j = Yc

        for j in range(0, g):
            u_j = Y_j[:, random.randint(0, self.Y_variables-1)]

            iteration_count = 0
            iteration_change = iteration_convergence * 10.0

            while iteration_count < max_iterations and \
                    iteration_change > iteration_convergence:

                t_j = K_j @ u_j
                t_j /= np.linalg.norm(t_j, 2)

                q_j = Y_j.T @ t_j

                old_u_j = u_j
                u_j = Y_j @ q_j
                u_j /= np.linalg.norm(u_j, 2)
                iteration_change = linalg.norm(u_j - old_u_j)
                iteration_count += 1

            if iteration_count >= max_iterations:
                if ignore_failures:
                    break
                else:
                    raise ConvergenceError('PLS2 failed to converge for '
                                           'component: '
                                           '{}'.format(self.components+1))

            T[:, j] = t_j
            Q[:, j] = q_j
            U[:, j] = u_j

            t_dot_t = t_j.T @ t_j
            P[:, j] = (K_j.T @ t_j) / t_dot_t
            tmp = (np.identity(self.data_samples) - np.outer(t_j.T, t_j))
            K_j = tmp @ K_j @ tmp
            Y_j = Y_j - np.outer(t_j, q_j.T)
            self.components += 1

        # If iteration stopped early because of failed convergence, only
        # the actual components will be copied

        self.T = T[:, 0:self.components]
        self.Q = Q[:, 0:self.components]
        self.U = U[:, 0:self.components]
        self.P = P[:, 0:self.components]

        self.B_RHS = self.U @ linalg.inv(self.T.T @ self.K @ self.U) @ self.Q.T

    def prediction(self, Z):
        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            Kt = np.empty((1, self.data_samples))
            Z = Z.reshape((1, Z.shape[0]))
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            Kt = np.empty((Z.shape[0], self.data_samples))

        for i in range(0, Z.shape[0]):
            for j in range(0, self.data_samples):
                Kt[i, j] = self.X_kernel(Z[i, :], self.X_training_set[j, :])

        centralizer = 1.0 / self.data_samples * \
            np.ones((Z.shape[0], self.data_samples))

        Kt = (Kt - centralizer @ self.K) @ \
            (np.identity(self.data_samples) - \
            1.0 / self.data_samples * np.ones(self.data_samples))

        return self.Y_offset + Kt @ self.B_RHS