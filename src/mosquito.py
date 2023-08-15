import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import trange

import arena


def turn(heading, turn_rate_sd):
    """
    Turn a heading in a random direction
    :param heading: current heading array
    :param turn_rate_sd: s.d. of turn rate from normal distribution
    """
    return heading + np.random.normal(0, turn_rate_sd, size=heading.shape)


def move(heading, speed):
    """
    Move a point in a given direction
    :param pos: current position (array)
    :param heading: current heading (array)
    :param speed: current speed (array)
    :return: new position (array)
    """
    return speed * np.array([np.cos(heading), np.sin(heading)]).T


def grad_move(pos, x_grad, y_grad, speed):
    """
    Move a point in the direction of the gradient
    :param pos: current position (Nx2 array)
    :param x_grad: x gradient (HxW array)
    :param y_grad: y gradient (HxW array)
    :param speed: current speed (Nx array)
    :return: movement (Nx2 array)
    """
    # Linear index of position
    idx = np.floor(pos[:, 1]).astype(int) * x_grad.shape[1] + np.floor(pos[:, 0]).astype(int)

    # Get gradient at current position
    x_move = x_grad.flat[idx] * speed.flatten()
    y_move = y_grad.flat[idx] * speed.flatten()

    # Calculate heading
    # heading = np.arctan2(y_grad, x_grad)

    return np.c_[x_move, y_move]


def is_at_site(pos, sites, site_density=None):
    """
    Check if a point is at a site
    :param pos: current position (Nx2 array)
    :param sites: site locations (HxW array)
    :return: boolean array of length N
    """
    # Linear index of position
    idx = np.floor(pos[:, 1]).astype(int) * sites.shape[1] + np.floor(pos[:, 0]).astype(int)

    # Check if at site
    if site_density is not None:
        passes = (sites.flat[idx] != 0) & (site_density.flat[idx] < np.random.uniform(1, 10, len(idx)))
        site_density.flat[idx[passes]] += 1
        return passes, site_density
    else:
        return sites.flat[idx] != 0


def isalive(pos, age, bounds):
    """
    Check if a point is alive (i.e. within the bounds)
    :param pos: current position (Nx2 array)
    :param bounds: bounds of the arena (2x2 array)
    :return: boolean array of length N
    """
    return np.all(pos > bounds[:, 0], axis=1) & np.all(pos < bounds[:, 1], axis=1) & (age < 100)


def plot(bounds, mosquitoes, trails, fed):
    """Plots mosquito trails and current position"""
    fig, ax = plt.subplots(figsize=(14.4, 10))
    # Setup axes
    ax.set_xlim(bounds[0, :])
    ax.set_ylim(bounds[1, :])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Show habitat + houses
    plt.imshow(np.ma.masked_where(arena.breed_distance > 10, arena.breed_distance), alpha=0.2)
    y, x = np.where(arena.feed_sites != 0)
    ax.scatter(x, y, color='brown', s=2, alpha=0.5)

    # Plot current position
    ax.scatter(mosquitoes[fed, 0], mosquitoes[fed, 1], color='r', s=2, alpha=0.5)
    ax.scatter(mosquitoes[~fed, 0], mosquitoes[~fed, 1], color='b', s=2, alpha=0.5)
    # Plot trails
    segs = trails.transpose((1, 0, 2))
    segs = np.ma.masked_where(segs == -999, segs)
    lines = LineCollection(segs, colors='k', alpha=0.2, linewidths=1)
    ax.add_collection(lines)

    fig.tight_layout()

    return fig, ax


if __name__ == '__main__':
    os.makedirs('../output', exist_ok=True)

    height, width = arena.feed_sites.shape
    bounds = np.array([[0, width], [0, height]])
    n = 100
    mosquitoes = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n, 2))
    headings = np.random.uniform(0, 2*np.pi, size=n)
    fed = np.zeros_like(headings).astype(bool)
    timer = np.zeros_like(headings)
    age = np.random.uniform(0, 50, size=fed.shape)

    site_density = np.zeros_like(arena.breed_sites, dtype=float)

    trails = np.copy([mosquitoes])
    trails_fed = np.copy([fed])

    grad_weight = 0.5

    for t in trange(500):
        alive = isalive(mosquitoes, age, bounds)
        timer[timer > 0] -= 1
        age[alive] += 1
        site_density[site_density > 0] -= 0.1
        site_density[site_density < 0] = 0
        n_mos = len(alive)

        # Turn & Move
        headings = turn(headings, np.pi/2)
        speed = np.fmax(0.01, np.random.normal(1, 0.5, size=(n_mos, 1)))
        heading_move = np.zeros_like(mosquitoes)
        heading_move[alive] = move(headings[alive], speed[alive])

        # Follow gradient
        feed_idx = alive & ~fed & (timer == 0)
        move_to_food = np.zeros_like(mosquitoes)
        move_to_food[feed_idx] = grad_move(mosquitoes[feed_idx], arena.feed_xgrad, arena.feed_ygrad, grad_weight * speed[feed_idx])

        breed_idx = alive & fed & (timer == 0)
        move_to_breed = np.zeros_like(mosquitoes)
        move_to_breed[breed_idx] = grad_move(mosquitoes[breed_idx], arena.breed_xgrad, arena.breed_ygrad, grad_weight * speed[breed_idx])

        # Combine
        mosquitoes += heading_move + move_to_food + move_to_breed
        alive = isalive(mosquitoes, age, bounds)

        # Check if mosquitoes have reached food/breeding site
        feed_idx = alive & ~fed & (timer == 0)  # recalculate as may have moved out of bounds
        can_feed = np.zeros_like(fed)
        can_feed[feed_idx] = is_at_site(mosquitoes[feed_idx], arena.feed_sites)

        breed_idx = alive & fed & (timer == 0)
        can_breed = np.zeros_like(fed)
        can_breed[breed_idx], site_density = is_at_site(mosquitoes[breed_idx], arena.breed_sites, site_density)

        fed[feed_idx & can_feed] = True
        fed[breed_idx & can_breed] = False

        timer[feed_idx & can_feed] = 10
        timer[breed_idx & can_breed] = 5

        # Create new mosquitoes
        new_mosquitoes = mosquitoes[can_breed].copy()
        new_mosquitoes = np.repeat(new_mosquitoes, 3, axis=0)

        n_new = new_mosquitoes.shape[0]
        new_headings = np.random.uniform(0, 2*np.pi, size=n_new)

        mosquitoes = np.concatenate([mosquitoes, new_mosquitoes], axis=0)
        headings = np.concatenate([headings, new_headings], axis=0)

        fed = np.concatenate([fed, np.zeros_like(new_headings).astype(bool)], axis=0)
        timer = np.concatenate([timer, np.zeros_like(new_headings)], axis=0)
        age = np.concatenate([age, np.zeros_like(new_headings)], axis=0)

        trails = np.pad(trails, ((0, 0), (0, n_new), (0, 0)), mode='constant', constant_values=-999)
        trails_fed = np.pad(trails_fed, ((0, 0), (0, n_new)), mode='constant', constant_values=0)

        # Records movement trail
        trails = np.concatenate([trails, [mosquitoes]], axis=0)
        trails_fed = np.vstack([trails_fed, fed])

        alive = isalive(mosquitoes, age, bounds)

        if False:
            print(n_new)
        else:
            fig, _ = plot(bounds, mosquitoes[alive], trails[:, alive, :], fed[alive])
            fig.savefig(f'../output/{t:03d}.png', dpi=400)
            plt.close(fig)

    # plot(bounds, mosquitoes[alive], trails[:, alive, :], fed[alive])
    plot(bounds, mosquitoes, trails, fed)
    plt.show()
    print(len(fed))