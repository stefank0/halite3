import logging

from hlt import constants
from scipy.optimize import linear_sum_assignment
#from matplotlib import pyplot as plt
import numpy as np
from utility import calc_distances, index_to_cell, cell_to_index
from schedule import Schedule

returning_to_dropoff = set()


def plot(arrs, fn):
    """Plot a dictionary of costs"""
    fig, axes = plt.subplots(len(arrs))
    for i, arr in enumerate(arrs):
        im = axes[i].imshow(arrs[arr])
        axes[i].set_title(arr)
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
        """Determine if ship has to return to a dropoff location"""
        if ship.id in returning_to_dropoff:
            return True
        ship_cell_index = cell_to_index(ship)
        dropoff = self.map_data.get_closest(ship, self.map_data.dropoffs)
        dropoff_index = cell_to_index(self.game_map[dropoff])
        if self.map_data.get_distance(ship_cell_index, dropoff_index) < 7:
            return ship.halite_amount > 0.75 * constants.MAX_HALITE
        else:
            return ship.halite_amount > 0.95 * constants.MAX_HALITE

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
        halite_density = self.map_data.halite_density
        threat_factor = 3.0 / (self.map_data.enemy_threat + 3.0)
        bonus_factor = 1.0 + 1.5 * (self.map_data.in_bonus_range > 1)
        c = -1.0 * halite_array * threat_factor * bonus_factor

        for i, ship in enumerate(remaining_ships):
            ship_cell_index = cell_to_index(ship)
            distance_array = self.map_data.get_distances(ship_cell_index)
            # Maximize the halite gathered per turn (considering only the first mining action)(factor 0.25 not
            # necessary, because it is there for all costs)(amount of turns: steps + 1 for mining)
            cost_matrix[i][:] = c / (distance_array + 1.0)
        return cost_matrix

    def assign_return(self, ship):
        """Assign this ship to return to closest dropoff."""
        returning_to_dropoff.add(ship.id)
        destination = self.map_data.get_closest(ship, self.map_data.dropoffs)
        ship_cell = self.game_map[ship]
        ship_cell_index = cell_to_index(ship)
        dropoff_index = cell_to_index(destination)
        # Create olifantenpaadjes:
        if (
            200 < self.turn_number < 300 and
            ship_cell.halite_amount > 40 and
            constants.MAX_HALITE - ship.halite_amount > 30 and
            self.map_data.get_distance(ship_cell_index, dropoff_index) < 7
        ):
            destination = ship.position
        self.schedule.assign(ship, destination)

    def to_destination(self):
        """Find the fit for the cost matrix"""
        for ship in self.ships:
            if ship.halite_amount < 0.25 * constants.MAX_HALITE:
                returning_to_dropoff.discard(ship.id)
            if self.map_data.free_turns(ship) < 7:
                returning_to_dropoff.add(ship.id)

        remaining_ships = []
        for ship in self.ships:
            if self.returning(ship):
                self.assign_return(ship)
            else:
                remaining_ships.append(ship)

        if self.dropoff_time(remaining_ships):
            ship = self.dropoff_ship(remaining_ships)
            if ship:
                self.schedule.dropoff(ship)
                remaining_ships.remove(ship)

        cost_matrix = self.create_cost_matrix(remaining_ships)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            ship = remaining_ships[i]
            destination = index_to_cell(j).position
            self.schedule.assign(ship, destination)
        return self.schedule

    def ship_matrix(self, ships):
        ship_matrix = np.ones(self.nmap)
        for ship in ships:
            ship_matrix[cell_to_index(ship)] = .5
        return ship_matrix

    def dropoff_time(self, ships):
        """Determine if it is time to create dropoff"""
        # TODO depends on number of ships as well
        dists = self.map_data.distance_dropoffs(ships)
        if dists.any() and (10 < self.turn_number < 0.8 * constants.MAX_TURNS):
            return dists.mean() > 10
        return False

    def dropoff_ship(self, ships):
        """Determine ship that creates dropoff"""
        # TODO depends on location of distance to enemy dropoffs as well
        ships = np.array(ships)
        halites = np.array([ship.halite_amount + self.game_map[ship].halite_amount for ship in ships])
        costs = constants.DROPOFF_COST - halites
        densities = np.array([self.map_data.halite_density[cell_to_index(ship)] for ship in ships])
        dists = self.map_data.distance_dropoffs(ships)
        suitable_ships = ships[(dists > 20) & (densities > 100) & (costs < self.me.halite_amount)]
        if len(suitable_ships) > 0:
            return suitable_ships[0]
        return False
