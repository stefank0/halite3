import logging

from hlt import constants
from scipy.optimize import linear_sum_assignment
#from matplotlib import pyplot as plt
import numpy as np
from utility import calc_distances, index_to_cell, cell_to_index
from schedule import Schedule

returning_to_shipyard = set()


def calc_shortest_dist(origin, destination):
    d_north, d_south, d_east, d_west = calc_distances(origin, destination)
    return min(d_north, d_south) + min(d_east, d_west)


def plot(costs, fn):
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


class Scheduler:
    """Creates a Schedule."""

    def __init__(self, _game, map_data):
        self.game_map = _game.game_map
        self.me = _game.me
        self.turn_number = _game.turn_number
        self.map_data = map_data
        self.schedule = Schedule(_game, map_data)
        self.ships = self.me.get_ships()
        self.nships = len(self.ships)
        self.nmap = self.game_map.width * self.game_map.height

    def get_schedule(self):
        self.to_destination()
        return self.schedule

    def returning(self, ship):
        if ship.id in returning_to_shipyard:
            return True
        ship_cell_index = cell_to_index(self.game_map[ship])
        shipyard_index = cell_to_index(self.game_map[self.me.shipyard])
        if self.map_data.get_distance(ship_cell_index, shipyard_index) < 7:
            return (ship.halite_amount > 0.75 * constants.MAX_HALITE)
        else:
            return (ship.halite_amount > 0.95 * constants.MAX_HALITE)

    def create_cost_matrix(self, remaining_ships):
        """Create a cost matrix for linear_sum_assignment() to determine the destination for each ship based on
        a combination of multiple costs matrices
        Note:
            The rows/columns of the cost matrix represent ships/targets. An
            element in the cost matrix represents the cost of moving a ship
            to a target. Some elements represent moves that are not possible
            in a single turn. However, because these have high costs, they will
            never be chosen by the algorithm.
        """

        cost_matrix = np.full((len(remaining_ships), self.nmap), 9999)
        halite_array = self.map_data.halite
        max_halite = np.max(halite_array)
        for i, ship in enumerate(remaining_ships):
            ship_cell_index = cell_to_index(self.game_map[ship])
            distance_array = self.map_data.get_distances(ship_cell_index)
            # Maximize the halite gathered per turn (considering only the first mining action)(factor 0.25 not necessary, because it is there for all costs)(amount of turns: steps + 1 for mining)
            cost_matrix[i][:] = -1.0 * halite_array / (distance_array + 1.0)
            """
            if i == 0 and self.turn_number in range(1, 100, 10):
                plot(costs={
                    'cost_matrix': cost_matrix[0][:].reshape(32, 32),
                    'halite_matrix': halite_matrix.reshape(32, 32),
                    'halite_matrix_cost': (np.max(np.sqrt(halite_matrix)) - np.sqrt(halite_matrix)).reshape(32, 32),
                    'dist_array_cost': dist_arr.reshape(32, 32),
                    'ship_matrix': ship_matrix.reshape(32, 32)
                }, fn=r'replays\img\ship_{}_turn_{}'.format(self.ships[0].id, self.turn_number))
            """

        return cost_matrix

    def to_destination(self):
        """Find the fit for the cost matrix"""
        for ship in self.ships:
            if ship.halite_amount < 0.25 * constants.MAX_HALITE:
                returning_to_shipyard.discard(ship.id)
            if self.map_data.free_turns(ship) < 7:
                returning_to_shipyard.add(ship.id)

        remaining_ships = []
        for ship in self.ships:
            if self.returning(ship):
                returning_to_shipyard.add(ship.id)
                destination = self.me.shipyard.position
                self.schedule.assign(ship, destination)
            else:
                remaining_ships.append(ship)

        cost_matrix = self.create_cost_matrix(remaining_ships)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            ship = remaining_ships[i]
            destination = index_to_cell(j).position
            logging.info('{}, {}'.format(ship, destination))
            self.schedule.assign(ship, destination)
        return self.schedule

    def ship_matrix(self, ships):
        ship_matrix = np.ones(self.nmap)
        for ship in ships:
            ship_matrix[cell_to_index(ship)] = .5
        return ship_matrix
