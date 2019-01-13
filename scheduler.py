import logging, math, time
from hlt import constants, entity
import numpy as np
from mapdata import to_cell, to_index, can_move, LinearSum, neighbours, simple_distance, enemy_ships
from schedule import Schedule
from parameters import param

returning_to_dropoff = set()


class Scheduler:
    """Creates a Schedule."""

    ghost = None

    @classmethod
    def spawn_ghost_dropoff(cls, map_data):
        """Instantiate a ghost dropoff, save it on the class."""
        cls.ghost = GhostDropoff(map_data)

    @classmethod
    def remove_ghost_dropoff(cls, map_data):
        """Remove the ghost dropoff from the class."""
        cls.ghost = None
        map_data.all_dropoffs = map_data.dropoffs

    def __init__(self, game, map_data):
        self.game_map = game.game_map
        self.me = game.me
        self.turn_number = game.turn_number
        self.turns_left = constants.MAX_TURNS - game.turn_number
        self.ships = self.me.get_ships()
        self.map_data = map_data
        self.schedule = Schedule(game, map_data)
        self.ships_per_dropoff = len(self.ships) / len(map_data.dropoffs)
        self.update_returning_to_dropoff()
        self.expected_halite = self._expected_halite()

    def mining_profit(self, bonussed_halite):
        """Calculate the total profit after mining up to 5 turns.

        Args:
            bonussed_halite (np.array): halite, including bonus factor.
        Returns:
            list(np.array): [<profit after 1 turn>, <profit after 2 turns>, ..]
        """
        multipliers = (0.25, 0.4375, 0.578125, 0.68359375, 0.7626953125)
        return [c * bonussed_halite for c in multipliers]

    def move_cost(self, halite):
        """Calculate the cost of moving after mining up to 3 turns.

        Args:
            halite (np.array): halite, not including bonus factor.
        Returns:
            list(np.array): [<cost after 1 turn>, <cost after 2 turns>, ..]
        """
        multipliers = (0.075, 0.05625, 0.0421875)
        return [c * halite for c in multipliers]

    def _neighbour_profit(self, profit):
        """Profit after mining up to 3 turns at the best neighbour cell.

        Note:
            Neighbour profit is capped by twice the profit on the cell itself.
        Args:
            profit (list(np.array)): result of mining_profit().
        Returns:
            list(np.array): [<profit after 1 turn>, <profit after 2 turns>, ..]
        """
        m = self.game_map.width * self.game_map.height
        profit_mining_once = profit[0]
        key = lambda index: profit_mining_once[index]
        best_neighbours = [max(neighbours(i), key=key) for i in range(m)]
        return [
            np.minimum(profit[0][best_neighbours], param['neighbour_profit_factor'] * profit[0]),
            np.minimum(profit[1][best_neighbours], param['neighbour_profit_factor'] * profit[1]),
            np.minimum(profit[2][best_neighbours], param['neighbour_profit_factor'] * profit[2])
        ]

    def multiple_turn_halite(self):
        """Max gathered halite within x turns, under some simple conditions.

        Reasoning:
            - Before, we used to maximize the average halite per turn up to and
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
            contribution cannot be much larger than the contribution of the
            cell itself, in order to avoid that neighbours of high halite cells
            always receive a high value. Implementation: the profit per turn on
            the neighbour is capped by twice the profit on the cell itself.
            - The first mining turn should be on the cell itself.
        Returns:
            list(np.array): [<maximum gathered halite after 1 turn>,
                             <maximum gathered halite after 2 turns>, ..]
        """
        halite = self.map_data.halite
        bonus_factor = 1 + (1 + param['extra_bonus']) * (self.map_data.in_bonus_range > 1)
        bonussed_halite = bonus_factor * halite
        profit = self.mining_profit(bonussed_halite)
        move_cost = self.move_cost(halite)
        neighbour_profit = self._neighbour_profit(profit)
        return self._find_max(profit, move_cost, neighbour_profit)

    def _find_max(self, profit, move_cost, neighbour_profit):
        """"Bookkeeping: find maximum halite gathered in x turns.

        Example:
            In 3 turns, the maximum halite gathered under the specified
            conditions is the max of the following two options:
            1) mining 3 times at the cell.
            2) mining once, moving to the best neighbour, mining once.
        Args:
            profit (list(np.array)): [<gathered halite after 1 turn>,
                                      <gathered halite after 2 turns>, ..]
            move_cost (list(np.array)): [<move cost after 1 turn>,
                                         <move cost after 2 turns>, ..]
            neighbour_profit (list(np.array)): Similar to profit.
        Returns:
            list(np.array): [<maximum gathered halite after 1 turn>,
                             <maximum gathered halite after 2 turns>, ..]
        """
        profit_mining_once_and_move = profit[0] - move_cost[0]
        profit_mining_twice_and_move = profit[1] - move_cost[1]
        profit_mining_thrice_and_move = profit[2] - move_cost[2]

        max_1turn = profit[0]
        max_2turns = profit[1]
        max_3turns = np.maximum(
            profit[2],
            profit_mining_once_and_move + neighbour_profit[0]
        )
        max_4turns = np.maximum.reduce([
            profit[3],
            profit_mining_twice_and_move + neighbour_profit[0],
            profit_mining_once_and_move + neighbour_profit[1]
        ])
        max_5turns = np.maximum.reduce([
            profit[4],
            profit_mining_thrice_and_move + neighbour_profit[0],
            profit_mining_twice_and_move + neighbour_profit[1],
            profit_mining_once_and_move + neighbour_profit[2]
        ])
        return [max_1turn, max_2turns, max_3turns, max_4turns, max_5turns]

    def valuable(self, ship, best_average_halite):
        """True if ship is expected to add a reasonable amount of halite."""
        expected_yield = best_average_halite * self.turns_left
        return (ship.halite_amount > 100 or
                ship.halite_amount + expected_yield > 200)

    def return_distances(self, ship):
        """Extra turns necessary to return to a dropoff."""
        dropoff_distances = self.map_data.calculator.simple_dropoff_distances
        dropoff_distance = dropoff_distances[to_index(ship)]
        return dropoff_distances - dropoff_distance

    def move_turns(self, ship, halite):
        """Turns spent on moving."""
        distances = self.map_data.get_distances(ship)
        return_distances = self.return_distances(ship)
        space = max(1, constants.MAX_HALITE - ship.halite_amount)
        move_turns = distances + param['return_distance_factor'] * (halite / space) * return_distances
        move_turns[move_turns < 0.0] = 0.0
        return move_turns

    def average(self, custom_mt_halite, ship):
        """Average halite gathered per turn (including movement turns)."""
        average_mt_halite = []
        for extra_turns, halite in enumerate(custom_mt_halite):
            mine_turns = 1.0 + extra_turns
            move_turns = self.move_turns(ship, halite)
            total_turns = mine_turns + move_turns
            halite[total_turns > self.turns_left] = 0.0
            average_mt_halite.append(halite / total_turns)
        return average_mt_halite

    def customize(self, mt_halite, ship):
        """Customize multiple_turn_halite for a specific ship."""
        space = constants.MAX_HALITE - ship.halite_amount
        custom_halite = [np.minimum(halite, space) for halite in mt_halite]
        loot = self.map_data.loot(ship)
        custom_halite[0] = np.maximum(custom_halite[0], loot)
        return custom_halite

    def initialize_cost_matrix(self, ships):
        """Ïnitialize the cost matrix with the correct shape."""
        m = self.game_map.width * self.game_map.height
        return np.zeros((len(ships), m))

    def create_cost_matrix(self, ships):
        """Cost matrix for linear_sum_assignment() to determine destinations."""
        mt_halite = self.multiple_turn_halite()

        cost_matrix = self.initialize_cost_matrix(ships)
        for i, ship in enumerate(ships):
            custom_mt_halite = self.customize(mt_halite, ship)
            average_mt_halite = self.average(custom_mt_halite, ship)
            best_average_halite = np.maximum.reduce(average_mt_halite)
            cost_matrix[i][:] = -1.0 * best_average_halite
        return cost_matrix

    def _kamikaze_cost(self, dropoff_index, ship_index, enemy_ship):
        """Cost value used to determine which enemy ship should be attacked."""
        i = to_index(enemy_ship)
        return simple_distance(dropoff_index, i) + simple_distance(ship_index, i)

    def assign_kamikaze(self, ship):
        """Attack with a ship that is no longer valuable, guard dropoffs."""
        dropoff = self.map_data.get_closest_dropoff(ship)
        dropoff_index = to_index(dropoff)
        ship_index = to_index(ship)
        possible_targets = list(enemy_ships())
        if possible_targets:
            target = min(possible_targets, key=lambda enemy_ship:
            self._kamikaze_cost(dropoff_index, ship_index, enemy_ship))
            self.schedule.assign(ship, target.position)
        else:
            self.schedule.assign(ship, ship.position)

    def _return_average_halite(self, ship):
        """Average returned halite per turn needed to return."""
        dropoff = self.map_data.get_closest_dropoff(ship)
        distance = self.map_data.get_entity_distance(ship, dropoff)
        return param['return_factor'] * ship.halite_amount / (2.0 * distance + 1.0)

    def assignment(self, ships):
        """Assign destinations to ships using an assignment algorithm."""
        cost_matrix = self.create_cost_matrix(ships)
        row_ind, col_ind = LinearSum.assignment(cost_matrix, ships)
        for i, j in zip(row_ind, col_ind):
            ship = ships[i]
            best_average_halite = -1.0 * cost_matrix[i, j]
            if not self.valuable(ship, best_average_halite):
                self.assign_kamikaze(ship)
            elif (ship.halite_amount > 550 and
                  self._return_average_halite(ship) > best_average_halite):
                self.assign_return(ship)
            else:
                destination = to_cell(j).position
                self.schedule.assign(ship, destination)

    def update_returning_to_dropoff(self):
        """Update the set of ships that are returning to a dropoff."""
        required_turns = math.ceil(self.ships_per_dropoff / 4.0) + 2
        for ship in self.ships:
            if ship.halite_amount < 0.25 * constants.MAX_HALITE:
                returning_to_dropoff.discard(ship.id)
            if self.map_data.free_turns(ship) < required_turns:
                returning_to_dropoff.add(ship.id)

    def assign_return(self, ship):
        """Assign this ship to return to closest dropoff."""
        returning_to_dropoff.add(ship.id)
        destination = self.map_data.get_closest_dropoff(ship)
        self.schedule.assign(ship, destination)

    def is_returning(self, ship):
        """Determine if ship has to return to a dropoff."""
        return (ship.id in returning_to_dropoff or
                ship.halite_amount > 0.95 * constants.MAX_HALITE)

    def preprocess(self, ships):
        """Process some ships in a specific way."""
        for ship in ships.copy():
            if not can_move(ship):
                self.schedule.assign(ship, ship.position)
                ships.remove(ship)
            elif self.is_returning(ship):
                self.assign_return(ship)
                ships.remove(ship)

    def get_schedule(self):
        """Create the Schedule, main method of Scheduler."""
        remaining_ships = self.ships.copy()
        self.dropoff_planning(remaining_ships)
        self.preprocess(remaining_ships)
        self.assignment(remaining_ships)
        return self.schedule

    def dropoff_planning(self, remaining_ships):
        """Handle dropoff planning and placement."""
        if self.is_dropoff_time():
            logging.info('is dropoff time')
            if self.ghost:
                ship = self.dropoff_ship()
                if ship:
                    self.schedule.dropoff(ship)
                    self.map_data.dropoffs.append(ship)
                    remaining_ships.remove(ship)
                    self.remove_ghost_dropoff(self.map_data)
                else:
                    self.ghost.move()
            else:
                if self.expected_halite > constants.DROPOFF_COST:
                    self.spawn_ghost_dropoff(self.map_data)
            if self.ghost and self.ghost.position is None:
                self.remove_ghost_dropoff(self.map_data)
        else:
            self.remove_ghost_dropoff(self.map_data)
        Scheduler.free_halite = self._free_halite()

    def is_dropoff_time(self):
        """Determine if it is time to create a dropoff."""
        end_game = self.turns_left < 0.2 * constants.MAX_TURNS
        early_game = self.turn_number < 150 and (len(self.map_data.dropoffs) == 1)
        return ((not end_game and self.ships_per_dropoff > 15) or
                (early_game and self.ships_per_dropoff > 10))

    def dropoff_cost(self, ship):
        """Cost of building a dropoff, taking into account reductions."""
        ship_halite = ship.halite_amount
        cell_halite = self.game_map[ship].halite_amount
        return max(constants.DROPOFF_COST - ship_halite - cell_halite, 0)

    def dropoff_ship(self):
        """Determine ship that creates the ghost dropoff."""
        for ship in self.ships:
            if ship.position == self.ghost.position:
                if (self.me.halite_amount < self.dropoff_cost(ship) or
                        to_cell(to_index(ship)).has_structure):
                    return None
                return ship
        return None

    def returning_ships(self):
        """List of ships that are returning to a dropoff."""
        ships = []
        for ship_id in returning_to_dropoff.copy():
            if self.me.has_ship(ship_id):
                ships.append(self.me.get_ship(ship_id))
            else:
                returning_to_dropoff.discard(ship_id)
        return ships

    def _delivers_in_time(self, ship, max_turns):
        """True if the ship delivers its cargo within max_turns."""
        destination = self.map_data.get_closest_dropoff(ship)
        distance = self.map_data.get_entity_distance(ship, destination)
        return distance < max_turns and destination != self.ghost

    def _expected_halite(self):
        """Halite expected to be available before the next dropoff creation."""
        max_turns = self.ghost.distance(self.ships) if self.ghost else 10
        returning_halite = 0.0
        for ship in self.returning_ships():
            if self._delivers_in_time(ship, max_turns):
                returning_halite += 0.8 * ship.halite_amount
        return self.me.halite_amount + returning_halite

    def _free_halite(self):
        """Calculate the halite that is free to be used for spawning ships."""
        halite = self.me.halite_amount
        if self.schedule.dropoff_assignments:
            ship = self.schedule.dropoff_assignments[0]
            return halite - self.dropoff_cost(ship)
        elif self.ghost:
            ship = entity.Ship(None, None, self.ghost.position, 500)
            return min(self.expected_halite - self.dropoff_cost(ship), halite)
        else:
            return halite


class GhostDropoff(entity.Entity):
    """Future dropoff, already taken into account by distance calculations."""

    search_radius1 = 15
    search_radius2 = 25

    def __init__(self, map_data):
        self.id = 999
        self.map_data = map_data
        self.calculator = map_data.calculator
        self.position = self.spawn_position()

    def _disputed_factor(self, index):
        """Gain a strategic advantage by controlling disputed areas."""
        d1 = self.calculator.enemy_dropoff_distances[index]
        d2 = self.calculator.simple_dropoff_distances[index]
        return min(1.1, max(1.0, 1.2 - 0.02 * abs(d1 - d2 - 2)))

    def _expansion_factor(self, index):
        """Reward gradual expansion."""
        d = self.calculator.simple_dropoff_distances[index]
        return max(1.0, 1.1 - 0.02 * abs(17.5 - d))

    def _turns(self, index):
        """Move turns uses Dijkstra distance of the second closest ship."""
        mine_turns = 10.0
        move_turns = self.calculator.ghost_distances[index]
        return mine_turns + min(30.0, max(15.0, move_turns))

    def cost(self, index):
        """Cost representing the quality of the index as a dropoff location.

        Note:
            Comparable to cost matrix of Scheduler, but with averages. This
            ensures that destinations of ships and dropoff placement match.
        """
        if (self.calculator.simple_dropoff_distances[index] < self.search_radius1 or
                self.map_data.density_difference[index] < 0.0 or
                self.map_data.halite_density[index] < 100.0 or
                to_cell(index).has_structure):
            return 0.0
        modifier = self._disputed_factor(index) * self._expansion_factor(index)
        halite_density = self.map_data.halite_density[index]
        return -1.0 * modifier * halite_density / self._turns(index)

    def spawn_positions(self):
        """Indices at which to search for spawn position."""
        r1 = self.search_radius1
        r2 = self.search_radius2
        d = self.calculator.simple_dropoff_distances
        return np.flatnonzero(np.logical_and(d >= r1, d <= r2)).tolist()

    def best_position(self, positions):
        """Determine the best position.

        Note:
            Returns None if a good position was not found. If that is the case,
            the GhostDropoff should not be considered any further.
        """
        spawn_index = min(positions, key=self.cost)
        if self.cost(spawn_index) == 0.0:
            return None
        return to_cell(spawn_index).position

    def spawn_position(self):
        """Determine a good position to 'spawn' the ghost dropoff."""
        positions = self.spawn_positions()
        return self.best_position(positions) if positions else None

    def distance(self, ships):
        """Simple distance of the current position to the nearest ship."""
        index = to_index(self)
        return min([simple_distance(index, to_index(ship)) for ship in ships])

    def move(self):
        """Move this ghost dropoff to a more optimal nearby location."""
        index = to_index(self)
        positions = (index,) + neighbours(index)
        self.position = self.best_position(positions)
