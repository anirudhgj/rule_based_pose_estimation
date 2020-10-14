import json
import logging

logger = logging.getLogger(__name__)

def debug_dict(data_dict, name=""):
    def get_str(k):
        if hasattr(data_dict[k], 'name') and hasattr(data_dict[k], 'shape'):
            return "{}: {}".format(data_dict[k].name, str(data_dict[k].shape))
        else:
            return str(data_dict[k])

    data_dict = {k: get_str(k) for k in data_dict}
    dict_str = json.dumps(data_dict, indent=4, sort_keys=True)
    logger.debug('%s : ' % name)
    logger.debug(dict_str)


import logging
import tensorflow as tf
import numpy as np
from scipy.stats import norm as sp_norm

logger = logging.getLogger(__name__)
# import tf.distributions.Normal as tf_normal

import matplotlib.pyplot as plt

def prob_dist(A, axis=-1):
#     A = np.array(A, dtype=np.float32)
#     assert all(A >= 0)
    return A / (tf.reduce_sum(A, axis=axis, keepdims=True) + 1e-9)


def unit_norm(A, axis=0):
#     A = np.array(A)
    norm = tf.linalg.norm(A, axis=axis, keepdims=True)
    return A / (norm + 1e-9)


def one_hot(labels, n_classes):
    return tf.eye(n_classes)[labels]


class DistTransform(object):
    def __init__(self, mu, sigma, low, high, scale=1.0):
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        self.scale = scale
        self.norm_for_cdf = tf.contrib.distributions.Normal(self.mu, self.sigma)
    def norm_to_uniform(self, data_norm):
        data_norm = data_norm * self.scale
        data_uniform = self.norm_for_cdf.cdf(data_norm)
        data_uniform = self.low + (self.high - self.low) * data_uniform
        data_uniform -= 1e-9 * tf.sign(data_uniform)
        return data_uniform

    def uniform_to_norm(self, data_uniform):
        data_uniform = (data_uniform - self.low) / (self.high - self.low)
        data_norm = self.norm_for_cdf.quantile(data_uniform)
        data_norm /= self.scale

        return data_norm


def convert_norm_to_uniform(zp_batch, dist_transform):
    zp_batch_uniform = dist_transform.norm_to_uniform(zp_batch)
    return zp_batch_uniform


def convert_uniform_to_norm(zp_batch_uniform, dist_transform):
    zp_batch_uniform = dist_transform.uniform_to_norm(zp_batch_uniform)
    return zp_batch_uniform