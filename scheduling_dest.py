import logging

import hlt
from hlt import Direction, Position
from scipy.optimize import linear_sum_assignment
import numpy as np
from scheduling import calc_distances, target, index_to_cell, cell_to_index, best_path, Schedule
from matplotlib import pyplot as plt


def calc_shortest_dist(origin, destination):
    d_north, d_south, d_east, d_west = calc_distances(origin, destination)
    return min(d_north, d_south) + min(d_east, d_west)


class Scheduler:
    """Keeps track of Assignments and translates them into a command list."""

    def __init__(self, _game_map, ships, turnnumber):
        self.schedule = Schedule(_game_map)
        self.game_map = _game_map
        self.ships = ships
        self.turnnumber = turnnumber
        self.n = len(self.ships)  # Number of assignments/ships.
        self.m = self.game_map.width * self.game_map.height  # Number of cells/targets.

    def initial_cost_matrix(self):
        """Initialize the cost matrix with high costs for every move.

        Note:
            The rows/columns of the cost matrix represent ships/targets. An
            element in the cost matrix represents the cost of moving a ship
            to a target. Some elements represent moves that are not possible
            in a single turn. However, because these have high costs, they will
            never be chosen by the algorithm.
        """
        return np.full((self.n, self.m), 9999)

    def create_cost_matrix(self):
        """"Create a cost matrix for linear_sum_assignment() to determine the destination for each ship"""
        cost_matrix = self.initial_cost_matrix()
        halite_matrix = self.halite_matrix()
        for i in range(len(self.ships)):
            dist_arr = self.schedule.dist_matrix[cell_to_index(self.ships[i])]
            cost_matrix[i][:] = (
                    np.max(np.sqrt(halite_matrix)) - np.sqrt(halite_matrix) +
                    dist_arr
            )
            if i == 0 and self.turnnumber in [2, 4, 6, 8]:
                self.plot(
                    costs={
                        'cost_matrix': cost_matrix[0][:].reshape(32, 32),
                        'halite_matrix': (np.max(np.sqrt(halite_matrix)) - np.sqrt(halite_matrix)).reshape(32, 32),
                        'dist_arr': dist_arr.reshape(32, 32),
                    },
                    fn=r'replays\img\ship_{}_turn_{}'.format(self.ships[0].id, self.turnnumber))
        return cost_matrix

    def plot(self, costs, fn):
        """Plot a dictionary of costs"""
        fig, axes = plt.subplots(len(costs))
        for i, cost in enumerate(costs):
            im = axes[i].imshow(costs[cost])
            axes[i].set_title(cost)
            fig.colorbar(im, ax=axes[i])
            axes[i].xaxis.set_major_formatter(plt.NullFormatter())
            axes[i].yaxis.set_major_formatter(plt.NullFormatter())
        fig.savefig(fn, bbox_inches='tight')
        return

    def halite_matrix(self):
        """ Create a 1D ndarray with halite"""
        halite_matrix = np.array(range(1000))
        # halite_matrix = cost_matrix.copy()[0][:]
        for j in np.indices(halite_matrix.shape)[0]:
            halite_matrix[j] = index_to_cell(j).halite_amount
        return halite_matrix

    def to_destination(self):
        """Find the fit for the cost matrix"""
        cost_matrix = self.create_cost_matrix()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            ship = self.ships[i]
            target = index_to_cell(j).position
            self.schedule.assign(ship, target)
            logging.info('Best destination is {} for ship {}'.format(ship, index_to_cell(j)))
        return self.schedule
