from hlt import Direction, Position
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import dijkstra, shortest_path
from scipy.sparse import csr_matrix
import numpy as np
import logging, math, time


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


def viable_directions(origin, destination):
    """Get a list of viable directions to get closer to the destination."""
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


def neighbours(index):
    """Return the indices of the neighbours of the cell belonging to index."""
    h = game_map.height
    w = game_map.width
    x = index % w
    y = (index // w)
    index_north = x + (w * ((y - 1) % h))
    index_south = x + (w * ((y + 1) % h))
    index_east = ((x + 1) % w) + (w * y)
    index_west = ((x - 1) % w) + (w * y)
    return index_north, index_south, index_east, index_west


def can_move(ship):
    """Return True if a ship is able to move."""
    necessary_halite = math.ceil(0.1 * game_map[ship].halite_amount)
    return necessary_halite <= ship.halite_amount


class Assignment:
    """An assignment of a ship to a destination."""

    def __init__(self, ship, destination):
        self.ship = ship
        self.destination = destination

    def targets(self):
        """Get a list of proper target cells for the next move."""
        origin = self.ship.position
        directions = viable_directions(origin, self.destination)
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

    edge_data = None

    def __init__(self, _game_map, _me):
        global game_map, me
        game_map = _game_map
        me = _me
        self.assignments = []
        self.halite = self.available_halite()
        self.graph = self.create_graph()
        self.dist_matrix, self.indices = self.shortest_path()

    def available_halite(self):
        """Get an array of available halite on the map."""
        m = game_map.height * game_map.width
        return np.array([index_to_cell(i).halite_amount for i in range(m)])

    def initialize_edge_data(self):
        """Store edge_data for create_graph() on the class for performance."""
        m = game_map.height * game_map.width
        col = np.array([j for i in range(m) for j in neighbours(i)])
        row = np.repeat(np.arange(m), 4)
        Schedule.edge_data = (row, col)

    def create_graph(self):
        """Create a matrix representing the game map graph.

        Note:
            The edge cost 1.0 + cell.halite_amount / 1000.0 is chosen such
            that the shortest path is mainly based on the number of steps
            necessary, but also slightly incorporates the halite costs of
            moving. Therefore, the most efficient path is chosen when there
            are several shortest distance paths.
        """
        if Schedule.edge_data is None:
            self.initialize_edge_data()
        edge_costs = np.repeat(1.0 + self.halite / 1000.0, 4)
        edge_data = Schedule.edge_data
        m = game_map.height * game_map.width
        return csr_matrix((edge_costs, edge_data), shape=(m, m))

    def shortest_path_indices(self):
        """Determine the indices for which to calculate the shortest path.

        Notes:
            - We also need the neighbours, because their results are used to
                generate the cost matrix for linear_sum_assignment().
            - list(set(a_list)) removes the duplicates from a_list.
        """
        indices = [cell_to_index(game_map[ship]) for ship in me.get_ships()]
        neighbour_indices = [j for i in indices for j in neighbours(i)]
        return list(set(indices + neighbour_indices))

    def shortest_path(self):
        """Calculate a perturbed distance from interesting cells to all cells.

        Possible performance improvements:
            - dijkstra's limit keyword argument.
            - reduce indices, for example by removing returning/mining ships.
            - reduce graph size, only include the relevant part of the map.
        """
        indices = self.shortest_path_indices()
        dist_matrix = dijkstra(self.graph, indices=indices)
        return dist_matrix, indices

    def get_distances(self, origin_index):
        """Get an array of perturbed distances from some origin cell."""
        return self.dist_matrix[self.indices.index(origin_index)]

    def get_distance(self, origin_index, target_index):
        """Get the perturbed distance from some cell to another."""
        return self.get_distances(origin_index)[target_index]

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
        return np.full((n, m), 99999999999.9)

    def reduce_feasible(self, cost_matrix):
        """Reduce the cost of all feasible moves for all ships."""
        for k, assignment in enumerate(self.assignments):
            ship = assignment.ship
            destination = assignment.destination
            origin_index = cell_to_index(game_map[ship])
            target_index = cell_to_index(game_map[assignment.destination])
            cost = self.get_distance(origin_index, target_index)
            cost_matrix[k][origin_index] = cost - 0.1
            if can_move(ship):
                for neighbour_index in neighbours(origin_index):
                    cost = self.get_distance(neighbour_index, target_index)
                    cost_matrix[k][neighbour_index] = cost

    def create_cost_matrix(self):
        """"Create a cost matrix for linear_sum_assignment()."""
        cost_matrix = self.initial_cost_matrix()
        self.reduce_feasible(cost_matrix)
        return cost_matrix

    def to_commands(self):
        """Translate the assignments of ships to commands."""
        commands = []
        cost_matrix = self.create_cost_matrix()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for k, i in zip(row_ind, col_ind):
            assignment = self.assignments[k]
            target = index_to_cell(i)
            commands.append(assignment.to_command(target))
        return commands
