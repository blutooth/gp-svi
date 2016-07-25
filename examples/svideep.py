from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from scipy.optimize import minimize


#2 layer, 1 dimension deep GP

def sparseGP(x):
    def unpack(x):
        u1_mean=
        u1_cov=
        z1=
        u2_mean=
        u2_cov=
        h_mean=
        h_cov=
        X=
        Y=
        kernel_noise=
        kernel_lenscale=
        function_noise=
     return




    def log_marginal_likelihood(self,params, x, y):
        mean, cov_params, noise_scale = self.rbf(params)
        cov_y_y = self.rbf(cov_params, x, x) + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y, True)

    def predict(self,params, x, y, xstar):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x, xstar)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov


    def rbf(kernel_params, x, xp):
        output_scale = np.exp(kernel_params[0])
        lengthscales = np.exp(kernel_params[1:])
        diffs = np.expand_dims(x / lengthscales, 1) \
                    - np.expand_dims(xp / lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def sample(self,mu,cov):

    def conditional(self,x,y,x_new):