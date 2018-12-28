import logging, math, time
import numpy as np

from hlt import Direction, constants
from mapdata import to_cell, to_index, can_move, neighbours, LinearSum, target, packing_fraction


class Assignment:
    """An assignment of a ship to a destination."""

    def __init__(self, ship, destination):
        self.ship = ship
        self.destination = destination

    def to_command(self, target_cell):
        """Return command to move its ship to a target cell."""
        target_cell.mark_unsafe(self.ship)

        if target_cell == game_map[self.ship]:
            return self.ship.stay_still()

        origin = self.ship.position
        for direction in Direction.get_all_cardinals():
            if target_cell == target(origin, direction):
                return self.ship.move(direction)


class Schedule:
    """Keeps track of Assignments and translates them into a command list."""

    def __init__(self, _game, map_data):
        global game, game_map, me
        game = _game
        game_map = game.game_map
        me = game.me
        self.assignments = []
        self.dropoff_assignments = []
        self.map_data = map_data
        self.calculator = map_data.calculator

    def assign(self, ship, destination):
        """Assign a ship to a destination."""
        assignment = Assignment(ship, destination)
        self.assignments.append(assignment)

    def dropoff(self, ship):
        """Assign a ship to become a dropoff."""
        self.dropoff_assignments.append(ship)

    def initial_cost_matrix(self):
        """Initialize the cost matrix with high costs for every move.

        Note:
            The rows/columns of the cost matrix represent ships/targets. An
            element in the cost matrix represents the cost of moving a ship
            to a target. Some elements represent moves that are not possible
            in a single turn. However, because these have high costs, they will
            never be chosen by the algorithm.
        """
        n = len(self.assignments)  # Number of assignments/ships.
        m = game_map.width * game_map.height  # Number of cells/targets.
        return np.full((n, m), 99999999999.9)

    def wasted_turn_cost(self, ship, target_index):
        """Costs (0.0 - 0.1) for a wasted turn. Also breaks some symmetry."""
        if to_index(ship) == target_index:
            return 0.0
        elif game_map[ship].has_structure:
            return 9999.0
        else:
            cargo_space = constants.MAX_HALITE - ship.halite_amount
            mining_potential = math.ceil(0.25 * game_map[ship].halite_amount)
            mining_profit = min(cargo_space, mining_potential)
            return min(0.0, 0.1 - 0.001 * mining_profit)

    def reduce_stay_still(self, cost_array, ship, destination):
        """Reduce the cost of staying still."""
        # Indices.
        target_index = to_index(game_map[destination])
        origin_index = to_index(ship)

        # Partial costs for the first turn (stay still).
        origin_threat = self.map_data.enemy_threat[origin_index]
        threat_cost = self.calculator.threat_func(ship, origin_threat)
        wasted_cost = self.wasted_turn_cost(ship, target_index)

        # Update cost matrix.
        first_cost = 1.0 + threat_cost + wasted_cost
        remaining_cost = self.map_data.get_distance(ship, target_index)
        cost_array[origin_index] = first_cost + remaining_cost

    def reduce_move(self, cost_array, ship, destination):
        """Reduce the cost of moving to a neighbouring cell."""
        # Indices.
        target_index = to_index(game_map[destination])
        origin_index = to_index(ship)

        for neighbour_index in neighbours(origin_index):
            # Update cost matrix.
            first_cost = self.map_data.get_distance(ship, neighbour_index)
            remaining_cost = self.calculator.get_distance_from_index(ship, neighbour_index, target_index)
            cost_array[neighbour_index] = first_cost + remaining_cost

    def reduce_feasible(self, cost_matrix):
        """Reduce the cost of all feasible moves for all ships.

        Note:
            The priority factor scales all values, and therefore it scales the
            differences between values. The larger the priority factor, the
            larger the differences between values and the more likely the
            algorithm chooses the most optimal value. The result is that when
            two ships want to move to the same location and their cost arrays
            are the same, the ship with higher priority factor gets priority.
        """
        for k, assignment in enumerate(self.assignments):
            ship = assignment.ship
            destination = assignment.destination
            cost_array = cost_matrix[k]
            self.reduce_stay_still(cost_array, ship, destination)
            if can_move(ship):
                self.reduce_move(cost_array, ship, destination)
            priority_factor = 1.0 + packing_fraction(ship)
            cost_matrix[k] = priority_factor * cost_array

    def create_cost_matrix(self):
        """"Create a cost matrix for linear_sum_assignment()."""
        cost_matrix = self.initial_cost_matrix()
        self.reduce_feasible(cost_matrix)
        return cost_matrix

    def to_commands(self):
        """Translate the assignments of ships to commands."""
        commands = []

        # Dropoff collisions.
        if self.allow_dropoff_collisions():
            self.resolve_dropoff_collisions(commands)

        # Assignment of next move.
        cost_matrix = self.create_cost_matrix()
        ships = [assignment.ship for assignment in self.assignments]
        row_ind, col_ind = LinearSum.assignment(cost_matrix, ships)
        for k, i in zip(row_ind, col_ind):
            assignment = self.assignments[k]
            target = to_cell(i)
            commands.append(assignment.to_command(target))

        # Create dropoff assignments.
        for ship in self.dropoff_assignments:
            commands.append(ship.make_dropoff())

        return commands

    def near_dropoff(self, ship):
        """Return True if the ship can reach a dropoff/shipyard this turn."""
        dropoff = self.map_data.get_closest_dropoff(ship)
        return to_index(ship) in neighbours(to_index(dropoff))

    def resolve_dropoff_collisions(self, commands):
        """Handle endgame collisions at closest dropoff."""
        remaining_assignments = []
        for assignment in self.assignments:
            ship = assignment.ship
            if self.near_dropoff(ship) and can_move(ship):
                dropoff = self.map_data.get_closest_dropoff(ship)
                commands.append(assignment.to_command(game_map[dropoff]))
            else:
                remaining_assignments.append(assignment)
        self.assignments = remaining_assignments

    def allow_dropoff_collisions(self):
        """True if endgame dropoff collisions are allowed."""
        ship_dropoff_ratio = len(me.get_ships()) / len(self.map_data.dropoffs)
        required_turns = math.ceil(ship_dropoff_ratio / 4.0) + 5
        turns_left = constants.MAX_TURNS - game.turn_number
        return turns_left <= required_turns
