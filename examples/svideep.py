
#2 layer, 1 dimension deep GP

class sparseGP():
    def __init__(self):
        self.u1_mean
        self.u1_cov
        self.z1
        self.u2_mean
        self.u2_cov
        self.h_mean
        self.h_cov
        self.X
        self.Y
        self.kernel_noise
        self.kernel_lenscale
        self.function_noise

    def prior(self,x,y):

    def sample(self,mu,cov):

    def conditional(self,x,y,x_new):