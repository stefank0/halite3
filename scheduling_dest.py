import logging

from hlt import Direction, Position
from scipy.optimize import linear_sum_assignment
import numpy as np
from scheduling import calc_distances, effective_directions, target, index_to_cell, cell_to_index, 


def calc_shortest_dist(origin, destination):
    d_north, d_south, d_east, d_west = calc_distances(origin, destination)
    return min(d_north, d_south) + min(d_east, d_west)


class Assignment:
    """An assignment of a ship to a destination."""

    def __init__(self, ship, destination):
        self.ship = ship
        self.destination = destination

    def targets(self):
        """Get a list of proper target cells for the next move."""
        origin = self.ship.position
        directions = effective_directions(origin, self.destination)
        return [target(origin, direction) for direction in directions]


class ScheduleDest:
    """Keeps track of Assignments and translates them into a command list."""

    def __init__(self, _game_map, ships:list):
        global game_map
        game_map = _game_map
        self.ships = ships
        self.assignments = []

    def initial_cost_matrix(self):
        """Initialize the cost matrix with high costs for every move.

        Note:
            The rows/columns of the cost matrix represent ships/targets. An
            element in the cost matrix represents the cost of moving a ship
            to a target. Some elements represent moves that are not possible
            in a single turn. However, because these have high costs, they will
            never be chosen by the algorithm.
        """

        n = len(self.ships)  # Number of assignments/ships.
        m = game_map.width * game_map.height  # Number of cells/targets.
        return np.full((n, m), 9999)

    def create_cost_matrix(self):
        """"Create a cost matrix for linear_sum_assignment()."""
        cost_matrix = self.initial_cost_matrix()
        halite_matrix = self.halite_matrix(cost_matrix)
        return cost_matrix

    def halite_matrix(self, cost_matrix):
        halite_matrix = cost_matrix.copy()[0][:]
        for j in np.indices(halite_matrix.shape)[0]:
            halite_matrix[j] = index_to_cell(j).halite_amount
        return halite_matrix

    def to_destination(self):
        """Find the fit for the cost matrix"""
        cost_matrix = self.create_cost_matrix()
        return cost_matrix
