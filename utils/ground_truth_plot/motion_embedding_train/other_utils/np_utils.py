import logging

import numpy as np
from scipy.stats import norm as sp_norm

logger = logging.getLogger(__name__)


def prob_dist(A, axis=-1):
    A = np.array(A, dtype=np.float32)
    assert all(A >= 0)
    return A / (A.sum(axis=axis, keepdims=True) + 1e-9)


def unit_norm(A, axis=0):
    A = np.array(A)
    norm = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / (norm + 1e-9)


def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


class DistTransform(object):
    def __init__(self, mu, sigma, low, high, scale=1.0):
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        self.scale = scale

    def norm_to_uniform(self, data_norm):
        data_norm = data_norm * self.scale
        data_uniform = sp_norm.cdf(data_norm, loc=self.mu, scale=self.sigma)
        data_uniform = self.low + (self.high - self.low) * data_uniform
        data_uniform -= 1e-9 * np.sign(data_uniform)
        return data_uniform

    def uniform_to_norm(self, data_uniform):
        data_uniform = (data_uniform - self.low) / (self.high - self.low)
        data_norm = sp_norm.ppf(data_uniform, loc=self.mu, scale=self.sigma)
        data_norm /= self.scale
        if np.isinf(data_norm).any():
            logger.error('DistTransform Error: for inf values, handling them')
            inf_mask = np.isinf(data_norm)
            data_norm[inf_mask] = np.sign(data_norm[inf_mask]) * 1.0
        return data_norm


def convert_norm_to_uniform(zp_batch, dist_transform):
    zp_batch_uniform = zp_batch.copy()
    zp_batch_uniform[:, :, :32] = dist_transform.norm_to_uniform(zp_batch[:, :, :32])
    zp_batch_uniform *= 10
    return zp_batch_uniform


def convert_uniform_to_norm(zp_batch_uniform, dist_transform):
    zp_batch = zp_batch_uniform.copy()
    zp_batch = zp_batch / 10
    zp_batch[:, :, :32] = dist_transform.uniform_to_norm(zp_batch[:, :, :32])
    return zp_batch