import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def turn(heading, turn_rate_sd):
    """
    Turn a heading in a random direction
    :param heading: current heading array
    :param turn_rate_sd: s.d. of turn rate from normal distribution
    """
    return heading + np.random.normal(0, turn_rate_sd, size=heading.shape)


def move(pos, heading, speed):
    """
    Move a point in a given direction
    :param pos: numpy array of shape (2,)
    :param heading: float
    :return: numpy array of shape (2,)
    """
    return pos + speed * np.array([np.cos(heading), np.sin(heading)]).T


if __name__ == '__main__':
    bounds = np.array([[-10, 10], [-10, 10]])
    n = 100
    mosquitoes = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n, 2))
    headings = np.random.uniform(0, 2*np.pi, size=n)

    trails = np.copy([mosquitoes])

    tt = np.array([[0, 0]])

    for t in range(8):
        headings = turn(headings, np.pi/2)
        speed = np.fmax(0.01, np.random.normal(1, 0.5, size=(n, 1)))
        mosquitoes = move(mosquitoes, headings, speed)
        trails = np.concatenate([trails, [mosquitoes]], axis=0)

        tt = np.append(tt, [mosquitoes[0, :]], axis=0)

    segs = [np.column_stack(
        (trails[:, i, 0], trails[:, i, 1]))
        for i in range(trails.shape[1])]

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
    lines = LineCollection(segs, colors='k', alpha=0.1, linewidths=1)
    ax.add_collection(lines)

    plt.show()
