from hlt import Direction, Position
from scipy.optimize import linear_sum_assignment
import numpy as np


# Oplossing bij einde spel, wanneer ze elkaar mogen raken op shipyard/dropoffs:
# - Extra artificial targets maken, zodat shipyards vaker gekozen kunnen worden.
# - Of de schepen grenzend aan een shipyard ertussenuit pikken en deze direct accepteren.

def calc_distances(origin, destination):
    """Calculates distances in all directions. Incorporates toroid metric."""
    dy = destination.y - origin.y
    dx = destination.x - origin.x
    height = game_map.height
    width = game_map.width
    d_south = dy if dy >= 0 else height + dy
    d_north = height - dy if dy >= 0 else -dy
    d_east = dx if dx >= 0 else width + dx
    d_west = width - dx if dx >= 0 else -dx
    return d_north, d_south, d_east, d_west


def effective_directions(origin, destination):
    """Get a list of effective directions to get closer to the destination."""
    directions = []
    (d_north, d_south, d_east, d_west) = calc_distances(origin, destination)
    if 0 < d_south <= d_north:
        directions.append(Direction.South)
    if 0 < d_north <= d_south:
        directions.append(Direction.North)
    if 0 < d_west <= d_east:
        directions.append(Direction.West)
    if 0 < d_east <= d_west:
        directions.append(Direction.East)
    if d_north == d_south == d_east == d_west == 0:
        directions.append(Direction.Still)
    return directions


def target(origin, direction):
    """Calculate the target cell if the ship moves in the given direction."""
    return game_map[origin.directional_offset(direction)]


def index_to_cell(index):
    """Map a 1D index to a 2D MapCell."""
    x = index % game_map.width
    y = index // game_map.width
    return game_map[Position(x, y)]


def cell_to_index(cell):
    """Map a 2D MapCell to a 1D index."""
    x = cell.position.x
    y = cell.position.y
    return x + game_map.width * y


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

    def __init__(self, _game_map):
        global game_map
        game_map = _game_map
        self.assignments = []

    def assign(self, ship, destination):
        """Assign a ship to a destination."""
        assignment = Assignment(ship, destination)
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
        return np.full((n, m), 9999)

    def reduce_stay_still(self, cost_matrix):
        """Not moving is a reasonable move for all ships."""
        for i, assignment in enumerate(self.assignments):
            current_cell = game_map[assignment.ship]
            j = cell_to_index(current_cell)
            cost_matrix[i][j] = 1

    def reduce_targets(self, cost_matrix):
        """The lowest costs are moves in the direction of the destination."""
        for i, assignment in enumerate(self.assignments):
            for target in assignment.targets():
                j = cell_to_index(target)
                cost_matrix[i][j] = 0

    def create_cost_matrix(self):
        """"Create a cost matrix for linear_sum_assignment()."""
        cost_matrix = self.initial_cost_matrix()
        self.reduce_stay_still(cost_matrix)
        self.reduce_targets(cost_matrix)
        return cost_matrix

    def to_commands(self):
        """Translate the assignments of ships to commands."""
        commands = []
        cost_matrix = self.create_cost_matrix()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, j in zip(row_ind, col_ind):
            assignment = self.assignments[i]
            target = index_to_cell(j)
            commands.append(assignment.to_command(target))
        return commands
