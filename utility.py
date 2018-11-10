from hlt import Direction, Position, constants
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from scipy.ndimage import uniform_filter
import numpy as np
import logging, math, time
from collections import Counter


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


def targets(origin, destination):
    """Get a list of proper target cells for the next move."""
    directions = viable_directions(origin, destination)
    return [target(origin, direction) for direction in directions]


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
    y = index // w
    index_north = x + (w * ((y - 1) % h))
    index_south = x + (w * ((y + 1) % h))
    index_east = ((x + 1) % w) + (w * y)
    index_west = ((x - 1) % w) + (w * y)
    return index_north, index_south, index_east, index_west


def bonus_neighbours(index):
    """Return a generator for the indices of the bonus neighbours.

    Note:
        A range of 3 is taken, so that the enemy ship cannot get out of range
        in the same turn and to make it more robust in general.
    """
    h = game_map.height
    w = game_map.width
    x = index % w
    y = index // w
    return (
        ((x + dx) % w) + (w * ((y + dy) % h))
        for dx in range(-3, 4)
        for dy in range(-3 + abs(dx), 4 - abs(dx))
    )


def ship_bonus_neighbours(ship):
    """Bonus neighbours for a ship."""
    ship_index = cell_to_index(game_map[ship])
    return bonus_neighbours(ship_index)


def threat(ship):
    """Get the indices threatened by an enemy ship.

    Note:
        The current location of the ship counts extra, because a ship is
        likely to stay still. Possible improvement: guess if the ship is going
        to move based on the halite of its current position and its cargo.
        At the moment, the ships current position is more threatening if it is
        not carrying much halite.
    """
    ship_index = cell_to_index(game_map[ship])
    if ship.halite_amount < 0.5 * constants.MAX_HALITE:
        factor = 12
    elif ship.halite_amount < 0.8 * constants.MAX_HALITE:
        factor = 6
    else:
        factor = 1
    return tuple(ship_index for i in range(factor)) + neighbours(ship_index)


def can_move(ship):
    """Return True if a ship is able to move."""
    necessary_halite = math.ceil(0.1 * game_map[ship].halite_amount)
    return necessary_halite <= ship.halite_amount


class MapData:
    """Analyzes the gamemap and provides useful data/statistics."""

    edge_data = None

    def __init__(self, _game):
        global game, game_map, me
        game = _game
        game_map = game.game_map
        me = game.me
        self.halite = self.available_halite()
        self.halite_density = self.density_available_halite()
        self.graph = self.create_graph()
        self.dist_matrix, self.indices = self.shortest_path()
        self.in_bonus_range = self.enemies_in_bonus_range()
        self.enemy_threat = self.calculate_enemy_threat()
        self.dropoffs = [me.shipyard] + me.get_dropoffs()

    def available_halite(self):
        """Get an array of available halite on the map."""
        m = game_map.height * game_map.width
        return np.array([index_to_cell(i).halite_amount for i in range(m)])

    def density_available_halite(self):
        """Get density of halite map with radius"""
        halite = self.halite.reshape(game_map.height, game_map.width)
        halite_density = uniform_filter(halite, size=9, mode='constant')
        return halite_density.ravel()

    def initialize_edge_data(self):
        """Store edge_data for create_graph() on the class for performance."""
        m = game_map.height * game_map.width
        col = np.array([j for i in range(m) for j in neighbours(i)])
        row = np.repeat(np.arange(m), 4)
        MapData.edge_data = (row, col)

    def create_graph(self):
        """Create a matrix representing the game map graph.

        Note:
            The edge cost 1.0 + cell.halite_amount / 750.0 is chosen such
            that the shortest path is mainly based on the number of steps
            necessary, but also slightly incorporates the halite costs of
            moving. Therefore, the most efficient path is chosen when there
            are several shortest distance paths.
            More solid justification: if mining yields 75 halite on average,
            one mining turn corresponds to moving over 75/(10%) = 750 halite.
            Therefore, moving over 1 halite corresponds to 1/750 of a turn.
        """
        if MapData.edge_data is None:
            self.initialize_edge_data()
        edge_costs = np.repeat(1.0 + self.halite / 750.0, 4)
        edge_data = MapData.edge_data
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
        dist_matrix = dijkstra(self.graph, indices=indices, limit=30.0)
        dist_matrix[dist_matrix == np.inf] = 99999.9
        return dist_matrix, indices

    def get_distances(self, origin_index):
        """Get an array of perturbed distances from some origin cell."""
        return self.dist_matrix[self.indices.index(origin_index)]

    def get_distance(self, origin_index, target_index):
        """Get the perturbed distance from some cell to another."""
        return self.get_distances(origin_index)[target_index]

    def get_closest(self, origin, dests):
        origin_index = cell_to_index(game_map[origin])
        dists = [self.get_distance(origin_index, cell_to_index(game_map[dest])) for dest in dests]
        return dests[dists.index(min(dists))]

    def free_turns(self, ship):
        """Get the number of turns that the ship can move freely."""
        ship_index = cell_to_index(game_map[ship])

        dropoff = self.get_closest(ship, self.dropoffs)
        dropoff_index = cell_to_index(game_map[dropoff])
        # shipyard_index = cell_to_index(game_map[me.shipyard])
        distance = self.get_distance(ship_index, dropoff_index)
        turns_left = constants.MAX_TURNS - game.turn_number
        return turns_left - math.ceil(distance)

    def _index_count(self, index_func):
        """Loops over enemy ships and counts indices return by index_func."""
        m = game_map.height * game_map.width
        index_count = np.zeros(m)
        temp = Counter(
            index
            for player in game.players.values() if player is not me
            for ship in player.get_ships()
            for index in index_func(ship)
        )
        for key, value in temp.items():
            index_count[key] = value
        return index_count

    def enemies_in_bonus_range(self):
        """Calculate the number of enemies within bonus range for all cells."""
        return self._index_count(ship_bonus_neighbours)

    def calculate_enemy_threat(self):
        """Calculate enemy threat for all cells."""
        return self._index_count(threat)

    def distance_dropoffs(self, ships):
        dists = []
        for ship in ships:
            dropoff = self.get_closest(ship, self.dropoffs)
            dists.append(self.get_distance(cell_to_index(ship), cell_to_index(dropoff)))
        return np.array(dists)
