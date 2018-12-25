import logging, math, time
from hlt import constants, entity
import numpy as np
from mapdata import to_cell, to_index, can_move, LinearSum, neighbours, simple_distance
from schedule import Schedule

returning_to_dropoff = set()
nothing_to_lose = set() # Can be used to defend or to attack most threatening player.


class Scheduler:
    """Creates a Schedule."""

    def __init__(self, _game, map_data):
        self.game = _game
        self.game_map = _game.game_map
        self.me = _game.me
        self.turn_number = _game.turn_number
        self.map_data = map_data
        self.schedule = Schedule(_game, map_data)
        self.ships = self.me.get_ships()
        self.nships = len(self.ships)
        self.nmap = self.game_map.width * self.game_map.height

    def returning(self, ship):
        """Determine if ship has to return to a dropoff."""
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

    def capped(self, halite, cargo_space):
        """Top off halite: a ship cannot mine more than it can carry."""
        return np.minimum(halite, cargo_space)

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

    def multiple_turn_halite(self):
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
            contribution cannot be much larger than the contribution of the
            cell itself, in order to avoid that neighbours of high halite cells
            always receive a high value. Implementation: the profit per turn on
            the neighbour is capped by twice the profit on the cell itself.
            - The first mining turn should be on the cell itself.
        Returns:
            list(np.array): [<maximum gathered halite after 1 turn>,
                             <maximum gathered halite after 2 turns>, ..]
        """
        m = self.nmap
        halite = self.map_data.halite
        bonus_factor = 1 + 2 * (self.map_data.in_bonus_range > 1)
        bonussed_halite = bonus_factor * halite
        profit = self.mining_profit(bonussed_halite)
        move_cost = self.move_cost(halite)
        reduced_profit = [
            profit[0] - move_cost[0],
            profit[1] - move_cost[1],
            profit[2] - move_cost[2]
        ]
        key = lambda index: bonussed_halite[index]
        best_neighbours = [max(neighbours(i), key=key) for i in range(m)]
        neighbour_profit = [
            np.minimum(profit[0][best_neighbours], 2 * profit[0]),
            np.minimum(profit[1][best_neighbours], 2 * profit[1]),
            np.minimum(profit[2][best_neighbours], 2 * profit[2])
        ]

        max_1turn = profit[0]
        max_2turns = profit[1]
        max_3turns = np.maximum(
            profit[2],
            reduced_profit[0] + neighbour_profit[0]
        )
        max_4turns = np.maximum.reduce([
            profit[3],
            reduced_profit[0] + neighbour_profit[1],
            reduced_profit[1] + neighbour_profit[0]
        ])
        max_5turns = np.maximum.reduce([
            profit[4],
            reduced_profit[0] + neighbour_profit[2],
            reduced_profit[1] + neighbour_profit[1],
            reduced_profit[2] + neighbour_profit[0]
        ])
        return [max_1turn, max_2turns, max_3turns, max_4turns, max_5turns]

    def return_distances(self, ship):
        """Get extra turns necessary to return to a dropoff."""
        dropoff_distances = self.map_data.calculator.simple_dropoff_distances
        dropoff_distance = dropoff_distances[to_index(ship)]
        return_distances = dropoff_distances - dropoff_distance
        return return_distances

    def create_cost_matrix(self, remaining_ships):
        """Cost matrix for linear_sum_assignment() to determine destinations.

        Note:
            The rows/columns of the cost matrix represent ships/targets. An
            element in the cost matrix represents the cost of moving a ship
            to a target. Some elements represent moves that are not possible
            in a single turn. However, because these have high costs, they will
            never be chosen by the algorithm.
        """
        cost_matrix = np.zeros((len(remaining_ships), self.nmap))
        halite = self.multiple_turn_halite()
        global_factor = self.map_data.global_factor
        turns_left = constants.MAX_TURNS - self.game.turn_number

        for i, ship in enumerate(remaining_ships):
            loot = self.map_data.loot(ship)
            cargo_space = constants.MAX_HALITE - ship.halite_amount
            distances = self.map_data.get_distances(ship)
            return_distances = self.return_distances(ship)

            # Top off halite based on cargo space and add loot.
            capped_halite = [self.capped(h, cargo_space) for h in halite]
            capped_halite[0] = np.maximum(capped_halite[0], loot)

            # Calculate the average halite gathered per turn.
            average_halite = []
            for extra_turns, h in enumerate(capped_halite):
                mine_turns = 1.0 + extra_turns
                move_turns = distances + (h / cargo_space) * return_distances
                move_turns[move_turns < 0.0] = 0.0
                total_turns = mine_turns + move_turns
                h[total_turns > turns_left] = 0.0
                average_halite.append(h / total_turns)

            # Maximize the average halite gathered per turn.
            best_average = np.maximum.reduce(average_halite)
            if best_average.max() * turns_left + ship.halite_amount > 50:
                cost_matrix[i][:] = -1.0 * global_factor * best_average
            else:
                nothing_to_lose.add(ship.id)
        return cost_matrix

    def assign_return(self, ship):
        """Assign this ship to return to closest dropoff."""
        returning_to_dropoff.add(ship.id)
        destination = self.map_data.get_closest_dropoff(ship)
        self.schedule.assign(ship, destination)

    def get_schedule(self):
        """Find the fit for the cost matrix"""
        required_turns = math.ceil(len(self.ships) / (4.0 * len(self.map_data.dropoffs)))
        for ship in self.ships:
            if ship.halite_amount < 0.25 * constants.MAX_HALITE:
                returning_to_dropoff.discard(ship.id)
            if self.map_data.free_turns(ship) < required_turns + 2:
                returning_to_dropoff.add(ship.id)

        returning_halite = self.returning_halite()
        remaining_ships = self.ships.copy()
        if self.dropoff_time(self.ships):
            if self.ghost:
                ship = self.dropoff_ship(remaining_ships)
                if ship:
                    self.remove_ghost_dropoff()
                    self.map_data.dropoffs.append(ship)
                    self.schedule.dropoff(ship)
                    remaining_ships.remove(ship)
                else:
                    self.ghost.move()
            else:
                if self.me.halite_amount + returning_halite > 4000:
                    self.spawn_ghost_dropoff(self.map_data)
        else:
            self.remove_ghost_dropoff()
        Scheduler.free_halite = self.calc_free_halite(returning_halite)

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

    def dropoff_time(self, ships):
        """Determine if it is time to create a dropoff."""
        ships_per_dropoff = len(ships) / len(self.map_data.dropoffs)
        crowded = ships_per_dropoff > 15
        end_game = self.turn_number > 0.8 * constants.MAX_TURNS
        return crowded and not end_game

    def dropoff_ship(self, ships):
        """Determine ship that creates the ghost dropoff."""
        for ship in ships:
            if ship.position == self.ghost.position:
                ship_halite = ship.halite_amount
                cell_halite = self.game_map[ship].halite_amount
                reduction = min(4000, cell_halite + ship_halite)
                if self.me.halite_amount < 4000 - reduction:
                    return None
                return ship
        return None

    ghost = None

    @classmethod
    def spawn_ghost_dropoff(cls, map_data):
        """Instantiate a ghost dropoff, save it on the class."""
        cls.ghost = GhostDropoff(map_data)
        if cls.ghost.position is None:
            cls.ghost = None

    @classmethod
    def remove_ghost_dropoff(cls):
        """Remove the ghost dropoff from the class."""
        cls.ghost = None

    def returning_ships(self):
        """List of ships that are returning to a dropoff."""
        ships = []
        for ship_id in returning_to_dropoff.copy():
            try:
                ships.append(self.me.get_ship(ship_id))
            except KeyError:
                returning_to_dropoff.discard(ship_id)
        return ships

    def returning_halite(self):
        """Halite that is on its way to be returned."""
        calculator = self.map_data.calculator
        max_distance = self.ghost.distance(self.ships) if self.ghost else 10.0
        halite = 0.0
        for ship in self.returning_ships():
            destination = self.map_data.get_closest_dropoff(ship)
            distance = calculator.get_entity_distance(ship, destination)
            if destination != self.ghost and distance < max_distance:
                halite += 0.8 * ship.halite_amount
        return halite

    def calc_free_halite(self, returning_halite):
        """Calculate the halite that is free to be used for spawning ships."""
        halite = self.me.halite_amount
        if self.schedule.dropoff_assignments:
            ship = self.schedule.dropoff_assignments[0]
            ship_halite = ship.halite_amount
            cell_halite = self.game_map[ship].halite_amount
            reduction = min(4000, cell_halite + ship_halite)
            return halite - (4000 - reduction)
        elif self.ghost:
            ship_halite = 500
            cell_halite = self.game_map[self.ghost].halite_amount
            reduction = min(4000, cell_halite + ship_halite)
            return min(halite, halite + returning_halite - (4000 - reduction))
        else:
            return halite


class GhostDropoff(entity.Entity):
    """Future dropoff already taken into account by distance functions."""

    search_radius1 = 17
    search_radius2 = 20  # To be optimized. Initial values based on watching some games by teccles.

    def __init__(self, map_data):
        self.map_data = map_data
        self.calculator = map_data.calculator
        self.position = self.spawn_position()

    def distance(self, ships):
        """Distance of the current position to the nearest ship."""
        index = to_index(self)
        return min([simple_distance(index, to_index(ship)) for ship in ships])

    def cost(self, index):
        """Cost representing the quality of the index as a dropoff location.

        TODO:
            - Add bonus for disputed areas (not clear who is going to take the
                halite from the area) to gain a strategic advantage.
                Use: simple_enemy_dropoff_distances
            - Add negative bonus for regions that are already taken by the
                enemy.
            - Add bonus when a dropoff on this location brings us closer to a
                large amount of far away halite.
        """
        cost = constants.DROPOFF_COST - to_cell(index).halite_amount - 750.0
        if self.map_data.density_difference[index] < 0.0:
            return 0.0
        if self.map_data.halite_density[index] < 100.0:
            return 0.0
        else:
            return -1.0 * self.map_data.halite_density[index]

    def search_path(self):
        """Indices at search_radius from the current dropoffs."""
        r1 = self.search_radius1
        r2 = self.search_radius2
        d = self.calculator.simple_dropoff_distances
        return np.flatnonzero(np.logical_or(d == r1, d == r2)).tolist()

    def spawn_position(self):
        """Determine a good position to 'spawn' the ghost dropoff."""
        search_path = self.search_path()
        if not search_path:
            return None
        spawn_index = min(search_path, key=self.cost)
        if self.cost(spawn_index) == 0.0:
            return None
        return to_cell(spawn_index).position

    def move(self):
        """Move this ghost dropoff to a more optimal nearby location.

        For example:
            Walk towards dropoffs if the neighbouring cells have too much
            halite, forcing returning ships to walk over them and to lose a lot
            of halite in returning.
            Walk away from enemies.
            Walk to a high value cell with low value neighbours (easy to get to
            and a lot of reduction of the build cost).
        """
        # TODO
        self.position = self.position
