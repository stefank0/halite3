from hlt import Direction, constants
import numpy as np
import logging, math, time
from mapdata import to_cell, to_index, can_move, neighbours, LinearSum, target


class Assignment:
    """An assignment of a ship to a destination."""

    def __init__(self, ship, destination):
        self.ship = ship
        self.destination = destination

    def to_command(self, target_cell):
        """Return command to move its ship to a target cell."""
        #if target_cell == game_map[self.ship] and game_map[self.destination] != game_map[self.ship]:
        #    logging.info('MOVE? {}) {} -> {} -> {}'.format(self.ship.id, self.ship.position, self.destination, target_cell.position))
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

    def assign(self, ship, destination):
        """Assign a ship to a destination."""
        assignment = Assignment(ship, destination)
        self.assignments.append(assignment)

    def dropoff(self, ship):
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

    def reduce_feasible(self, cost_matrix):
        """Reduce the cost of all feasible moves for all ships."""
        for k, assignment in enumerate(self.assignments):
            ship = assignment.ship
            destination = assignment.destination
            origin_index = to_index(ship)
            target_index = to_index(game_map[destination])
            origin_enemy_cost = self.map_data.enemy_threat[origin_index]
            edge_to_self_cost = 1.0 + self.map_data.calculator.threat_costs_func(ship, origin_enemy_cost)
            remaining_cost = self.map_data.get_distance(ship, target_index)
            cost_matrix[k][origin_index] = edge_to_self_cost + remaining_cost
            #logging.info("{}) {} - {}".format(ship.id, edge_to_self_cost, remaining_cost))
            if can_move(ship):
                for neighbour_index in neighbours(origin_index):
                    first_edge_cost = self.map_data.get_distance(ship, neighbour_index)
                    remaining_cost = self.map_data.calculator.get_distance_from_index(
                        ship, neighbour_index, target_index
                    )
                    #logging.info("{}) {} - {}".format(ship.id, first_edge_cost, remaining_cost))
                    cost_matrix[k][neighbour_index] = first_edge_cost + remaining_cost

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
        ships = [assignment.ship for assignment in self.assignments]
        row_ind, col_ind = LinearSum.assignment(cost_matrix, ships)
        for k, i in zip(row_ind, col_ind):
            assignment = self.assignments[k]
            target = to_cell(i)
            commands.append(assignment.to_command(target))
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
            dropoff = self.map_data.get_closest_dropoff(assignment.ship)
            dropoff_cell = game_map[dropoff]
            if self.near_dropoff(assignment.ship) and can_move(assignment.ship):
                commands.append(assignment.to_command(dropoff_cell))
            else:
                remaining_assignments.append(assignment)
        self.assignments = remaining_assignments

    def allow_dropoff_collisions(self):
        """Return True if we allow endgame dropoff collisions at a closest dropoff."""
        turns_left = constants.MAX_TURNS - game.turn_number
        required_turns = math.ceil(len(me.get_ships()) / (4.0 * len(self.map_data.dropoffs))) + 2
        return turns_left <= required_turns
