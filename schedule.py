from hlt import Direction, constants
from scipy.optimize import linear_sum_assignment
import numpy as np
import logging, math, time
from utility import target, index_to_cell, cell_to_index, can_move, neighbours


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


class AssignmentDropoff(Assignment):
    """An assignment of a ship to make a dropoff."""

    def __init__(self, ship):
        Assignment.__init__(self, ship, ship.position)

    def to_command(self, target_cell):
        target_cell.mark_unsafe(self.ship)
        return self.ship.make_dropoff()


class Schedule:
    """Keeps track of Assignments and translates them into a command list."""

    def __init__(self, _game, map_data):
        global game, game_map, me
        game = _game
        game_map = game.game_map
        me = game.me
        self.assignments = []
        self.map_data = map_data

    def assign(self, ship, destination):
        """Assign a ship to a destination."""
        assignment = Assignment(ship, destination)
        self.assignments.append(assignment)

    def dropoff(self, ship):
        assignment = AssignmentDropoff(ship)
        self.assignments.append(assignment)

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

    def reduce_feasible(self, cost_matrix):
        """Reduce the cost of all feasible moves for all ships."""
        for k, assignment in enumerate(self.assignments):
            ship = assignment.ship
            destination = assignment.destination
            origin_index = cell_to_index(game_map[ship])
            target_index = cell_to_index(game_map[destination])
            cost = self.map_data.get_distance(origin_index, target_index)
            cost_matrix[k][origin_index] = cost - 0.1
            if can_move(ship):
                for neighbour_index in neighbours(origin_index):
                    cost = self.map_data.get_distance(neighbour_index, target_index)
                    cost_matrix[k][neighbour_index] = cost

    def create_cost_matrix(self):
        """"Create a cost matrix for linear_sum_assignment()."""
        cost_matrix = self.initial_cost_matrix()
        self.reduce_feasible(cost_matrix)
        return cost_matrix

    def to_commands(self):
        """Translate the assignments of ships to commands."""
        commands = []
        if self.allow_dropoff_collisions():
            self.resolve_dropoff_collisions(commands)
        cost_matrix = self.create_cost_matrix()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for k, i in zip(row_ind, col_ind):
            assignment = self.assignments[k]
            target = index_to_cell(i)
            commands.append(assignment.to_command(target))
        return commands

    def near_dropoff(self, ship):
        """Return True if the ship can reach a dropoff/shipyard this turn."""
        ship_index = cell_to_index(game_map[ship])
        for dropoff in self.map_data.dropoffs:
            dropoff_index = cell_to_index(dropoff)
            if ship_index in neighbours(dropoff_index):
                return True
        return False

    def resolve_dropoff_collisions(self, commands):
        """Handle endgame collisions at closest dropoff."""
        remaining_assignments = []
        for assignment in self.assignments:
            dropoff = self.map_data.get_closest(assignment.ship, self.map_data.dropoffs)
            dropoff_cell = game_map[dropoff]
            if self.near_dropoff(assignment.ship):
                commands.append(assignment.to_command(dropoff_cell))
            else:
                remaining_assignments.append(assignment)
        self.assignments = remaining_assignments

    def allow_dropoff_collisions(self):
        """Return True if we allow endgame dropoff collisions at a closest dropoff."""
        turns_left = constants.MAX_TURNS - game.turn_number
        ships_left = len(me.get_ships())
        return turns_left <= math.ceil(ships_left / 4.0)
