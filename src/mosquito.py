import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

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


def isalive(pos, bounds):
    """
    Check if a point is alive (i.e. within the bounds)
    :param pos: current position (Nx2 array)
    :param bounds: bounds of the arena (2x2 array)
    :return: boolean array of length N
    """
    return np.logical_and(np.all(pos > bounds[:, 0], axis=1), np.all(pos < bounds[:, 1], axis=1))


def plot(bounds, mosquitoes, trails):
    """Plots mosquito trails and current position"""
    fig, ax = plt.subplots()
    # Setup axes
    ax.set_xlim(bounds[0, :])
    ax.set_ylim(bounds[1, :])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot current position
    ax.scatter(mosquitoes[:, 0], mosquitoes[:, 1], color='k', s=2, alpha=0.5)
    # Plot trails
    lines = LineCollection(trails.transpose((1, 0, 2)), colors='k', alpha=0.1, linewidths=1)
    ax.add_collection(lines)

    plt.show()


if __name__ == '__main__':
    height, width = arena.feed_sites.shape
    bounds = np.array([[0, width], [0, height]])
    n = 100
    mosquitoes = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n, 2))
    headings = np.random.uniform(0, 2*np.pi, size=n)

    trails = np.copy([mosquitoes])

    for t in range(100):
        alive = isalive(mosquitoes, bounds)
        old_pos = mosquitoes.copy()

        # Turn & Move
        headings = turn(headings, np.pi/2)
        speed = np.fmax(0.01, np.random.normal(1, 0.5, size=(n, 1)))
        heading_move = move(headings, speed)

        # Follow gradient
        move_to_food = np.zeros_like(mosquitoes)
        move_to_food[alive] = grad_move(mosquitoes[alive], arena.feed_xgrad, arena.feed_ygrad, speed[alive])

        # Combine
        mosquitoes += heading_move + move_to_food

        # Records movement trail
        trails = np.concatenate([trails, [mosquitoes]], axis=0)

    plot(bounds, mosquitoes, trails)
