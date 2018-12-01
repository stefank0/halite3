import logging, math, time
from hlt import constants
import numpy as np
from mapdata import to_cell, to_index, can_move, LinearSum, neighbours
from schedule import Schedule

returning_to_dropoff = set()


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
        dropoff = self.map_data.get_closest_dropoff(ship)
        dropoff_index = to_index(dropoff)
        if self.map_data.get_distance(ship, dropoff_index) < 7:
            return ship.halite_amount > 0.75 * constants.MAX_HALITE
        else:
            return ship.halite_amount > 0.95 * constants.MAX_HALITE

    def remove_exhausted(self, halite):
        """Only keep cells that have a reasonable amount of halite.

        Experimental feature.
        Desired behaviour:
            When there are no reasonable cells left close to the ship, for
            example immediately after a dropoff, make sure the ship does not
            choose a bad nearby cell. Instead, choose a more distant one.
            Hopefully, this has the side-effect that Olifantenpaadjes are
            created, because bad cells that are not on a shortest path are
            left alone completely and the shortest path is mined even further.
        """
        threshold = np.mean(halite) - 0.5 * np.std(halite)
        halite[halite < threshold] = 0.0

    def capped(self, halite, ship):
        """Top off halite: a ship cannot mine more than it can carry."""
        cargo_space = constants.MAX_HALITE - ship.halite_amount
        return np.minimum(halite, 4.0 * cargo_space)

    def mining_profit(bonussed_halite):
        """Calculate the total profit after mining up to 5 turns.

        Args:
            bonussed_halite (np.array): halite, including bonus factor.
        Returns:
            list(np.array): [<profit after 1 turn>, <profit after 2 turns>, ..]
        """
        multipliers = (0.25, 0.4375, 0.578125, 0.68359375, 0.7626953125)
        return [c * bonussed_halite for c in multipliers]

    def move_cost(halite):
        """Calculate the cost of moving after mining up to 3 turns.

        Args:
            halite (np.array): halite, not including bonus factor.
        Returns:
            list(np.array): [<cost after 1 turn>, <cost after 2 turns>, ..]
        """
        multipliers = (0.075, 0.05625, 0.0421875)
        return [c * halite for c in multipliers]

    def multiple_turn_halite():
        """Max gathered halite within x turns, under some simple conditions.

        Reasoning:
            - Currently, we maximize the average halite per turn up to and
            including the next mining turn. Turtles should not be this greedy
            and plan a little bit further ahead. It is feasible to calculate
            the maximum halite minable within the next x turns, for small x,
            which is done by this method. This information is then used to see
            if the average halite per turn can be greater if we consider a
            couple of mining turns at once.
            - Bonus factor is 3 instead of 2, because you also take halite from
            the enemy, by taking it first.
        Conditions:
            - A single neighbouring cell is allowed to contribute, but this
            contribution cannot be larger than the contribution of the cell
            itself, in order to avoid that neighbours of high halite cells
            always receive a high value.
            - The first mining turn should be on the cell itself.
        """
        m = self.nmap
        halite = self.map_data.halite
        bonus_factor = 1 + 2 * (self.map_data.in_bonus_range > 1)
        bonussed_halite = bonus_factor * halite
        mining_profit = self.mining_profit(bonussed_halite)
        move_cost = self.move_cost(halite)
        key = lambda index: bonussed_halite[index]
        best_neighbours = [max(neighbours(i), key=key) for i in range(m)]
        neighbour_mining_profit = mining_profit[best_neighbours]

        # Implement first condition.

        max1 = mining_profit[0]
        max2 = mining_profit[1]
        max3 = np.maximum(
            mining_profit[2],
            mining_profit[0] + move_cost[0] + neighbour_mining_profit[0]
        )
        max4 = np.maximum.reduce([
            mining_profit[3],
            mining_profit[0] + move_cost[0] + neighbour_mining_profit[1],
            mining_profit[1] + move_cost[1] + neighbour_mining_profit[0]
        ])
        max5 = np.maximum.reduce([
            mining_profit[4],
            mining_profit[0] + move_cost[0] + neighbour_mining_profit[2],
            mining_profit[1] + move_cost[1] + neighbour_mining_profit[1],
            mining_profit[2] + move_cost[2] + neighbour_mining_profit[0]
        ])
        return max1, max2, max3, max4, max5

    def create_cost_matrix(self, remaining_ships):
        """Cost matrix for linear_sum_assignment() to determine destinations.

        Note:
            The rows/columns of the cost matrix represent ships/targets. An
            element in the cost matrix represents the cost of moving a ship
            to a target. Some elements represent moves that are not possible
            in a single turn. However, because these have high costs, they will
            never be chosen by the algorithm.
        """

        cost_matrix = np.full((len(remaining_ships), self.nmap), 9999)
        halite_array = self.map_data.halite
        global_threat_factor = self.map_data.global_threat
        bonus_factor = 1 + 3 * (self.map_data.in_bonus_range > 1)
        apparent_halite = halite_array * global_threat_factor
        self.remove_exhausted(apparent_halite)
        apparent_halite *= bonus_factor

        for i, ship in enumerate(remaining_ships):
            loot = self.map_data.loot(ship)
            halite = self.capped(apparent_halite + loot, ship)
            distance_array = self.map_data.get_distances(ship)
            # Maximize the halite gathered per turn (considering only the first mining action)(factor 0.25 not
            # necessary, because it is there for all costs)(amount of turns: steps + 1 for mining)
            cost_matrix[i][:] = -1.0 * halite / (distance_array + 1.0)
        return cost_matrix

    def assign_return(self, ship):
        """Assign this ship to return to closest dropoff."""
        returning_to_dropoff.add(ship.id)
        destination = self.map_data.get_closest_dropoff(ship)
        dropoff_index = to_index(destination)
        # Create olifantenpaadjes:
        if (
            200 < self.turn_number < 300 and
            self.game_map[ship].halite_amount > 40 and
            constants.MAX_HALITE - ship.halite_amount > 30 and
            self.map_data.get_distance(ship, dropoff_index) < 7
        ):
            destination = ship.position
        self.schedule.assign(ship, destination)

    def to_destination(self):
        """Find the fit for the cost matrix"""
        required_turns = math.ceil(len(self.ships) / (4.0 * len(self.map_data.dropoffs)))
        for ship in self.ships:
            if ship.halite_amount < 0.25 * constants.MAX_HALITE:
                returning_to_dropoff.discard(ship.id)
            if self.map_data.free_turns(ship) < required_turns + 2:
                returning_to_dropoff.add(ship.id)

        remaining_ships = self.ships.copy()
        if self.dropoff_time(remaining_ships):
            ship = self.dropoff_ship(remaining_ships)
            if ship:
                self.schedule.dropoff(ship)
                remaining_ships.remove(ship)

        for ship in remaining_ships[:]:
            if self.returning(ship):
                self.assign_return(ship)
                remaining_ships.remove(ship)
            elif not can_move(ship):
                self.schedule.assign(ship, ship.position)
                remaining_ships.remove(ship)
        cost_matrix = self.create_cost_matrix(remaining_ships)
        row_ind, col_ind = LinearSum.assignment(cost_matrix, remaining_ships)
        for i, j in zip(row_ind, col_ind):
            ship = remaining_ships[i]
            destination = to_cell(j).position
            self.schedule.assign(ship, destination)
        return self.schedule

    def ship_matrix(self, ships):
        ship_matrix = np.ones(self.nmap)
        for ship in ships:
            ship_matrix[to_index(ship)] = .5
        return ship_matrix

    def dropoff_time(self, ships):
        """Determine if it is time to create dropoff"""
        dists = self.map_data.distance_dropoffs(ships)
        ship_per_dropoff = len(ships) / len(self.map_data.dropoffs)
        return (
                dists.any() and
                dists.mean() > 10 and
                (self.turn_number < 0.8 * constants.MAX_TURNS) and
                ship_per_dropoff > 10
        )

    def dropoff_ship(self, ships):
        """Determine ship that creates dropoff"""
        halites = np.array([ship.halite_amount + self.game_map[ship].halite_amount for ship in ships])
        costs = constants.DROPOFF_COST - halites
        halite_densities = np.array([self.map_data.halite_density[to_index(ship)] for ship in ships])
        ship_densities = np.array([self.map_data.ship_density[to_index(ship)] for ship in ships])
        dists = self.map_data.distance_dropoffs(ships)
        suitability = (
                halite_densities *
                (dists > 10) *
                (costs < self.me.halite_amount) *
                (halite_densities > 100) *
                (ship_densities > 0)
        )
        best_ship_id = suitability.argsort()[-1]
        if suitability[best_ship_id]:
            self.map_data.dropoffs.append(ships[best_ship_id])
            return ships[best_ship_id]
        return False
