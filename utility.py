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


def simple_distance(index_a, index_b):
    """"Get the actual step distance from one cell to another."""
    height = game_map.height
    width = game_map.width
    dx = abs((index_b % width) - (index_a % width))
    dy = abs((index_b // width) - (index_a // width))
    return min(dx, width - dx) + min(dy, height - dy)


def simple_distances(index):
    """Get an array of the actual step distances to all cells."""
    m = game_map.height * game_map.width
    return np.array([simple_distance(index, i) for i in range(m)])


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


def to_cell(index):
    """Map a 1D index to a 2D MapCell."""
    x = index % game_map.width
    y = index // game_map.width
    return game_map[Position(x, y)]


def to_index(obj):
    """Map a 2D MapCell or Entity to a 1D index."""
    return obj.position.x + game_map.width * obj.position.y


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
    """Return a generator for the indices of the bonus neighbours."""
    h = game_map.height
    w = game_map.width
    x = index % w
    y = index // w
    return (
        ((x + dx) % w) + (w * ((y + dy) % h))
        for dx in range(-4, 5)
        for dy in range(-4 + abs(dx), 5 - abs(dx))
    )


def ship_bonus_neighbours(ship):
    """Bonus neighbours for a ship."""
    ship_index = to_index(ship)
    return bonus_neighbours(ship_index)


def threatening_ships(ship):
    """Return a list of nearby enemy ships."""
    h = game_map.height
    w = game_map.width
    x = ship.position.x
    y = ship.position.y
    nearby_cells = (
        game_map._cells[(y + dy) % h][(x + dx) % w]
        for dx in range(-2, 3)
        for dy in range(-2 + abs(dx), 3 - abs(dx))
        if not dx == dy == 0
    )
    return [
        cell.ship
        for cell in nearby_cells
        if cell.is_occupied and cell.ship.owner is not me
    ]


def threat(ship):
    """Get the indices threatened by an enemy ship.

    Note:
        The current location of the ship counts extra, because a ship is
        likely to stay still. Possible improvement: guess if the ship is going
        to move based on the halite of its current position and its cargo.
        At the moment, the ships current position is more threatening if it is
        not carrying much halite.
    """
    ship_index = to_index(ship)
    factor = math.ceil(4.0 * (1.0 - packing_fraction(ship)**2))
    return tuple(ship_index for i in range(factor)) + neighbours(ship_index)


def can_move(ship):
    """Return True if a ship is able to move."""
    necessary_halite = math.ceil(0.1 * game_map[ship].halite_amount)
    return necessary_halite <= ship.halite_amount


def packing_fraction(ship):
    """Get the packing/fill fraction of the ship."""
    return ship.halite_amount / constants.MAX_HALITE


class MapData:
    """Analyzes the gamemap and provides useful data/statistics."""

    edge_data = None

    def __init__(self, _game):
        global game, game_map, me
        game = _game
        game_map = game.game_map
        me = game.me
        self.dropoffs = [me.shipyard] + me.get_dropoffs()
        self.halite = self.get_available_halite()
        self.occupied = self.get_occupation()
        self.total_halite = self.get_total_halite()
        self.halite_density = self.density_available_halite()
        if MapData.edge_data is None:
            self.initialize_edge_data()
        self.dropoff_cost = self.dropoff_distance_edge_cost()
        self._dist_matrices = self.shortest_path()
        self.in_bonus_range = self.enemies_in_bonus_range()
        self.global_threat = self.calculate_global_threat()

    def get_available_halite(self):
        """Get an array of available halite on the map."""
        m = game_map.height * game_map.width
        return np.array([to_cell(i).halite_amount for i in range(m)])

    def get_total_halite(self):
        """Get an array of available halite, including enemy cargo."""
        halite = np.copy(self.halite)
        for player in game.players.values():
            if player is not me:
                for ship in player.get_ships():
                    ship_index = to_index(ship)
                    halite[ship_index] += ship.halite_amount
        return halite

    def density_available_halite(self):
        """Get density of halite map with radius"""
        halite = self.halite.reshape(game_map.height, game_map.width)
        halite_density = uniform_filter(halite, size=9, mode='constant')
        return halite_density.ravel()

    def get_occupation(self):
        """Get an array describing occupied cells on the map."""
        m = game_map.height * game_map.width
        return np.array([
            to_cell(j).is_occupied
            for i in range(m) for j in neighbours(i)
        ])

    def initialize_edge_data(self):
        """Store edge_data for create_graph() on the class for performance."""
        m = game_map.height * game_map.width
        col = np.array([j for i in range(m) for j in neighbours(i)])
        row = np.repeat(np.arange(m), 4)
        MapData.edge_data = (row, col)

    def create_graph(self, ship):
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
            The term self.occupied is added, so that the shortest path also
            takes traffic delays into consideration.
            The term packing_fraction(ship) * self.dropoff_cost represents
            costs for turns needed to return to a dropoff. If the ship is
            almost full, only the next mining action benefits from the move
            that increased the distance. Therefore, the extra turns needed to
            return are added to the costs. However, when the ship is almost
            empty, many mining turns benefit from the move and therefore the
            extra turns are only slightly added to the costs.
        """
        edge_costs = np.repeat(1.0 + self.halite / 750.0, 4) + self.occupied + packing_fraction(ship) * self.dropoff_cost
        edge_data = MapData.edge_data
        m = game_map.height * game_map.width
        return csr_matrix((edge_costs, edge_data), shape=(m, m))

    def _ship_shortest_path(self, ship):
        """Calculate shortest path from a ship to all cells."""
        graph = self.create_graph(ship)
        ship_index = to_index(ship)
        indices = (ship_index, ) + neighbours(ship_index)
        dist_matrix = dijkstra(graph, indices=indices, limit=30.0)
        dist_matrix[dist_matrix == np.inf] = 99999.9
        return dist_matrix, indices

    def shortest_path(self):
        """Calculate shortest paths for all ships."""
        dist_matrices = {}
        for ship in me.get_ships():
            dist_matrices[ship.id] = self._ship_shortest_path(ship)
        return dist_matrices

    def get_distance_from_index(self, ship, from_index, to_index):
        """Get the distance from index (near ship) to index."""
        dist_matrix, indices = self._dist_matrices[ship.id]
        return dist_matrix[indices.index(from_index)][to_index]

    def get_distances(self, ship):
        """Get an array of perturbed distances to all cells."""
        dist_matrix, _indices = self._dist_matrices[ship.id]
        return dist_matrix[0]

    def get_distance(self, ship, index):
        """Get the perturbed distance from a ship an index (a cell)."""
        return self.get_distances(ship)[index]

    def get_entity_distance(self, ship, entity):
        """"Get the perturbed distance from a ship to an Entity."""
        return self.get_distance(ship, to_index(entity))

    def get_closest(self, ship, destinations):
        """Get the destination that is closest to the ship."""
        key = lambda destination: self.get_entity_distance(ship, destination)
        return min(destinations, key=key)

    def get_closest_dropoff(self, ship):
        """Get the dropoff that is closest to the ship."""
        return self.get_closest(ship, self.dropoffs)

    def free_turns(self, ship):
        """Get the number of turns that the ship can move freely."""
        dropoff = self.get_closest(ship, self.dropoffs)
        distance = self.get_entity_distance(ship, dropoff)
        turns_left = constants.MAX_TURNS - game.turn_number
        return turns_left - math.ceil(distance)

    def mining_probability(self, ship):
        """Estimate the probability that a ship will mine the next turn."""
        if not can_move(ship):
            return 1.0
        ship_index = to_index(ship)
        simple_cost = self.halite / (simple_distances(ship_index) + 1.0)
        mining_cost = simple_cost[ship_index]
        moving_cost = np.delete(simple_cost, ship_index).max()
        cargo_factor = min(1.0, 10.0 * (1.0 - packing_fraction(ship)))
        return cargo_factor * mining_cost / (mining_cost + moving_cost)

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

    def distance_dropoffs(self, ships):
        """Get a list of distances to the nearest dropoff for all ships."""
        return np.array([
            self.get_entity_distance(ship, self.get_closest_dropoff(ship))
            for ship in ships
        ])

    def simple_dropoff_distances(self):
        """Simple step distances from all cells to the nearest dropoff."""
        distances_to_all_dropoffs = np.array([
            simple_distances(to_index(dropoff)) for dropoff in self.dropoffs
        ])
        return np.min(distances_to_all_dropoffs, axis=0)

    def dropoff_distance_edge_cost(self):
        """Edge costs representing increased dropoff distance.

        Costs:
            Increased distance = 1.0
            Equal distance = 0.5
            Decreased distance = 0.0
        """
        dropoff_distances = self.simple_dropoff_distances()
        row, col = MapData.edge_data
        return 0.5 * (dropoff_distances[col] - dropoff_distances[row] + 1.0)

    def calculate_global_threat(self):
        """Calculate enemy threat factor for all cells."""
        return 3.0 / (self._index_count(threat) + 3.0)

    def local_threat(self, ship):
        """Calculate enemy threat factor near a ship.

        Strategy:
            Take into account the amount of collisions with the enemy player:
            - Keep track of how and to whom you lost your own ships.
            - Flee/attack more aggresively for aggresive players (tit-for-tat).
        """
        m = game_map.height * game_map.width
        threat = np.ones(m)
        for enemy_ship in threatening_ships(ship):
            mining_probability = self.mining_probability(enemy_ship)
            dhalite = ship.halite_amount - enemy_ship.halite_amount
        return threat
