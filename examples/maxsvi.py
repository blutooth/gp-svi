from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
import gaussian_process as gp
import autograd.scipy.stats.multivariate_normal as mvn


from autograd import grad
from optimizers import adam



def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:2*D]
        inputs=np.reshape(params[2*D:3*D],(D,1))
        len_sc, variance = params[3*D], params[3*D+1]
        meany=mean*inputs
        return mean, log_std, inputs, len_sc, variance

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std,inputs, len_sc, variance = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        print(log_std)
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples,inputs,len_sc,variance, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params



if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-posterior.
    D = 10;dim=1;N=20;M=10;
    params = [0, - 6.32795237, - 0.69221531, - 0.24707744]

    # Build params = [0, - 6.32795237, - 0.69221531, - 0.24707744]model and objective function.
    num_params, predict, log_marginal_likelihood = \
        gp.make_gp_funs(gp.rbf_covariance, num_cov_params=D + 1)

    X, y = gp.build_toy_dataset(D=dim,n_data=N)
    pseudo, blah = gp.build_toy_dataset(D=dim,n_data=M)
    out,blah=gp.build_toy_dataset(D=dim,n_data=100)

    objective = lambda params: -log_marginal_likelihood(params, X, y)

    def log_posterior(x,inputs ,len_sc,variance, t):
        N=x.shape
        sum_prob=0
        params = [0, len_sc, variance, - 0.24707744]
        for i in range(N[0]):

            """An example 2D intractable distribution:
            a Gaussian evaluated at zero with a Gaussian prior on the log-variance."""
            mu= x[i][:]



            pred_mean, pred_cov = predict(params, inputs, mu, X)
            prior           = log_marginal_likelihood(params,inputs,mu)
            posterior       = mvn.logpdf(y, pred_mean, pred_cov, True)
        sum_prob=posterior+sum_prob+prior
        return sum_prob

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_posterior, D, num_samples=1000)

    # Set up plotting code
    def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z)
        ax.set_yticks([])
        ax.set_xticks([])

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params,t,g):

        paramis = [0, - 6.32795237, - 0.69221531, - 0.24707744]
        print("Log likelihood {}".format(-objective(params,t)))
        plt.cla()
        # Show posterior marginals.
        plot_xs = np.reshape(np.linspace(-7, 7, 300), (300, 1))
        mu, log_std = params[:D], params[D:2*D]
        pred_mean, pred_cov = predict(paramis, pseudo, mu, plot_xs)
        marg_std = np.sqrt(np.diag(pred_cov))
        ax.plot(plot_xs, pred_mean, 'b')
        ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
        np.concatenate([pred_mean - 1.96 * marg_std,
                                  (pred_mean + 1.96 * marg_std)[::-1]]),
                 alpha=.15, fc='Blue', ec='None')
        plt.pause(1.0/30.0)
        ax.plot(X, y, 'kx')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.draw()
        plt.pause(1.0 / 60.0)

        print("Iteration {} lower bound {}".format(t, -objective(params, t)))

    print("Optimizing variational parameters...")
    init_mean    = -1 * np.ones(D)
    init_log_std = -5 * np.ones(D)
    z=pseudo
    init_var_params = np.concatenate([init_mean, init_log_std,np.reshape(z,(D)),[1],[1]])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=500, callback=callback)