from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import os

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
    os.makedirs('../figures', exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(np.ma.masked_where(feed_sites == 0, feed_sites)))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/feed_sites.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(np.ma.masked_where(breed_sites == 0, breed_sites)))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/breed_sites.png', dpi=300)
    plt.show()



    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(breed_distance))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/breed_distance.png', dpi=300)
    plt.show()


    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(breed_xgrad))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/breed_xgrad.png', dpi=300)
    plt.show()


    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(breed_ygrad))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/breed_ygrad.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(feed_distance))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/feed_distance.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(feed_xgrad))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/feed_xgrad.png', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(14.4, 10))
    ax.imshow(np.flipud(feed_ygrad))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('../figures/feed_ygrad.png', dpi=300)
    plt.show()

    # plt.imshow(feed_xgrad)
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(feed_ygrad)
    # plt.colorbar()
    # plt.show()
