from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np


def load_data(fname):
    data = loadmat(fname)
    locations = (data['arr'] != 0).astype(int)
    distmap = data['distmap']

    # Calculate distance gradient
    y_grad, x_grad = np.gradient(distmap)

    return locations, distmap, -x_grad, y_grad


feed_sites, feed_distance, feed_xgrad, feed_ygrad = load_data('../data/feed_sites.mat')
breed_sites, breed_distance, breed_xgrad, breed_ygrad  = load_data('../data/breed_sites.mat')


if __name__ == '__main__':
    plt.imshow(feed_sites)
    plt.colorbar()
    plt.show()

    plt.imshow(feed_distance)
    plt.colorbar()
    plt.show()

    plt.imshow(feed_xgrad)
    plt.colorbar()
    plt.show()

    plt.imshow(feed_ygrad)
    plt.colorbar()
    plt.show()
