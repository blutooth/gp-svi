from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from scipy.optimize import minimize
from autograd import grad
import scipy as sp

#2 layer, 1 dimension deep GP
n = 200
m = 10
X=np.random.randn(n, 1)
Y=np.random.randn(n, 1)
"""
"""""


def initParams(num):

    mat = np.random.randn(m, m)
    return dict({num + 'z': np.reshape(np.linspace(0.0, 1.0, num=m), (m, 1)),
            num + 'u_mean': np.random.randn(m, 1),
            num + 'u_cov_fac': mat @ mat.T,
            num + 'h_mean': np.random.randn(n, 1),
            num + 'h_cov_fac': np.random.randn(n, 1),
            num + 'kernel_noise': np.ones((1, 1)),
            num + 'kernel_lenscale': np.ones((1, 1)),
            num + 'function_noise': np.ones((1, 1))})



def black_box_variational_inference(logprob, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_cov(params):
        return params['cov_params']

    def unpack_params(params, num):
        return params[num + 'u_mean'],params[num + 'u_cov_fac'],params[num + 'h_mean'],params[num + 'h_cov_fac']


    rs = npr.RandomState(0)
    def variational_objective(params):
        """Provides a stochastic estimate of the variational lower bound."""
        u1_mean, u1_cov_fac,h_mean, h_cov_fac = unpack_params(params,'1')
        u2_mean,u2_cov_fac,_,_ = unpack_params(params,'2')


        log_prob=0
        for i in range(num_samples):
            sample_u1 = u1_cov_fac @npr.rand(m,1)  + u1_mean
            sample_u2 = u2_cov_fac @ npr.rand(m,1) + u2_mean
            sample_h = h_cov_fac*h_cov_fac* npr.rand(n,1) + h_mean

            log_prob=log_prob+logprob(params,sample_u1,X,sample_h,'1')
            log_prob=log_prob+logprob(params,sample_u2,sample_h,Y,'2')
        log_prob=log_prob/num_samples
        #entropy=mvn.entropy(np.reshape(u1_mean,-1), (u1_cov_fac @ u1_cov_fac.T))
        #entropy=entropy-mvn.entropy(np.reshape(u2_mean,-1), (u2_cov_fac @ u2_cov_fac.T))
        #entropy=entropy-mvn.entropy(np.reshape(h_mean,-1), np.diag(h_mean) @ np.diag(h_mean))
        print(log_prob)#+entropy)

        return -(log_prob)#+entropy)

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def layer_bound(params,u,h_in,h_out,num):

    tol=1e-4
    def RBF(x,xp):
        output_scale = params[num+'kernel_noise']
        lengthscales = params[num+'kernel_lenscale']
        diffs =  np.expand_dims(x / lengthscales, 1) - np.expand_dims(xp / lengthscales, 0)
        return  output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    noise_scale=params[num+'function_noise']
    z=params[num+'z']

    def log_marginal(x, y):
        cov_y_y = RBF(x, x) + noise_scale * np.eye(len(y))
        return np.sum(mvn.logpdf(y, np.zeros(len(y)), cov_y_y, True))

    def conditional(x, y, xstar):
        #Assume zero mean prior
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise).
           -assumed prior mean is zero mean u is the observed"""
        cov_f_f = RBF(xstar, xstar)
        cov_y_f = RBF(x, xstar)
        cov_y_y = RBF(x, x) + (noise_scale+tol) * np.eye(len(y))
        pred_mean = np.dot(solve(cov_y_y, cov_y_f).T, y)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)+tol*np.eye(len(xstar))
        return pred_mean, pred_cov

    def log_cond(x,y,xstar,ystar):
        pred_mean, pred_cov= conditional(x,y,xstar)
        return np.sum(mvn.logpdf(ystar,np.reshape(pred_mean,-1),pred_cov))


    def sample(shape,mu,cov):
        return mu+np.linalg.cholesky(cov) @ np.randn(shape)

    return log_cond(z,u,h_in,h_out)+log_marginal(z,u)



if __name__ == '__main__':
    layer1_params=initParams('1')
    layer2_params=initParams('2')

    init_params={**layer1_params, **layer2_params}
    variational_objective, gradient, unpack_params= black_box_variational_inference(layer_bound, 5)


print(variational_objective(init_params),gradient(init_params))


    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(params):
        print("Log likelihood {}".format(-objective(params)))
        plt.cla()
        print(params)
        # Show posterior marginals.
        plot_xs = np.reshape(np.linspace(-7, 7, 300), (300,1))
        pred_mean, pred_cov = predict(params, X, y, plot_xs)
        marg_std = np.sqrt(np.diag(pred_cov))
        ax.plot(plot_xs, pred_mean, 'b')
        ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
                np.concatenate([pred_mean - 1.96 * marg_std,
                               (pred_mean + 1.96 * marg_std)[::-1]]),
                alpha=.15, fc='Blue', ec='None')

        # Show samples from posterior.
        rs = npr.RandomState(0)
        sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov, size=10)
        ax.plot(plot_xs, sampled_funcs.T)

        ax.plot(X, y, 'kx')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.draw()
        plt.pause(1.0/60.0)

    # Initialize covariance parameters
    rs = npr.RandomState(0)
    init_params = 0.1 * rs.randn(num_params)

    print("Optimizing covariance parameters...")
    cov_params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='CG', callback=callback)
    plt.pause(10.0)
