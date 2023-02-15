import numpy as np
from bisect import bisect_left

def get_sample_from_multi_gaussian(lambda_,gamma_,mean):
    # Here lambda and gamma are the eigen decomposition of the corresponding covariance matrix
    dimensions = len(lambda_)
    # sampling from normal distribution
    x_normal = np.random.randn(dimensions)
    # transforming into multivariate distribution
    x_multi = (x_normal*lambda_) @ gamma_ + mean
    return x_multi

def get_samples_from_mixed_gaussian(c,means,variances,num_samples):
    n = len(c)
    accum = np.zeros(n)
    accum[0] = c[0]
    for i in range(1,n):
        accum[i] = accum[i-1]+c[i]
    lambdas = []
    gamma = []
    for i in range(n):
        lambda_, gamma_ = np.linalg.eig(np.array(variances[i]))
        lambdas.append(lambda_)
        gamma.append(gamma_)
    samples = []
    for i in range(num_samples):
        idx = bisect_left(accum,np.random.rand(1)[0])
        samples.append(get_sample_from_multi_gaussian(lambda_=lambdas[idx],gamma_=gamma[idx],mean=means[idx]))
    samples = np.array(samples)
    return samples

