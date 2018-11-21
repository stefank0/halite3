from hlt import Position, constants
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from collections import Counter
import numpy as np
import logging, math, time


##############################################################################
#
# Utility
#
##############################################################################


def to_index(obj):
    """Map a 2D MapCell or Entity to a 1D index."""
    return obj.position.x + game_map.width * obj.position.y


def to_cell(index):
    """Map a 1D index to a 2D MapCell."""
    x = index % game_map.width
    y = index // game_map.width
    return game_map[Position(x, y)]


def can_move(ship):
    """True if a ship is able to move."""
    necessary_halite = math.ceil(0.1 * game_map[ship].halite_amount)
    return necessary_halite <= ship.halite_amount


def packing_fraction(ship):
    """Get the packing/fill fraction of the ship."""
    return ship.halite_amount / constants.MAX_HALITE


def neighbours(index):
    """Get the indices of the neighbours of the cell belonging to index."""
    h = game_map.height
    w = game_map.width
    x = index % w
    y = index // w
    index_north = x + (w * ((y - 1) % h))
    index_south = x + (w * ((y + 1) % h))
    index_east = ((x + 1) % w) + (w * y)
    index_west = ((x - 1) % w) + (w * y)
    return index_north, index_south, index_east, index_west


def neighbourhood(index, radius):
    """Generator for all indices around index within a radius."""
    h = game_map.height
    w = game_map.width
    x = index % w
    y = index // w
    return (
        ((x + dx) % w) + (w * ((y + dy) % h))
        for dx in range(-radius, radius + 1)
        for dy in range(-radius + abs(dx), radius + 1 - abs(dx))
    )


def calc_density(radius, array, count_self=True):
    """Calculate a density map based on a radius (sum of density and array remain the same)."""
    density = np.zeros(array.shape)
    if count_self:
        density = array.copy() / (radius + 1)
    for dist in range(1, radius + 1):
        x = dist
        while x >= 1:
            density += np.roll(np.roll(array,  dist - x, 0),  x, 1) / ((radius + 1) * dist * 4)  # southeast
            density += np.roll(np.roll(array,  dist - x, 1), -x, 0) / ((radius + 1) * dist * 4)  # northeast
            density += np.roll(np.roll(array, -dist + x, 0), -x, 1) / ((radius + 1) * dist * 4)  # northwest
            density += np.roll(np.roll(array, -dist + x, 1),  x, 0) / ((radius + 1) * dist * 4)  # southwest
            x += -1
    return density


##############################################################################
#
# Distances
#
##############################################################################


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


class DistanceCalculator:
    """Calculates shortest path distances for all ships."""

    _edge_data = None

    def __init__(self, halite, dropoffs, enemy_threat):
        if DistanceCalculator._edge_data is None:
            self._initialize_edge_data()
        self.dropoffs = dropoffs
        self.simple_dropoff_distances = self._simple_dropoff_distances(dropoffs)
        self._traffic_costs = self._traffic_edge_costs()
        self._movement_costs = self._movement_edge_costs(halite)
        self._return_costs = self._return_edge_costs(dropoffs)
        self._threat_costs = self._threat_edge_costs(enemy_threat)
        self._dist_tuples = self._shortest_path()

    def _initialize_edge_data(self):
        """Store edge_data for create_graph() on the class for performance."""
        m = game_map.height * game_map.width
        col = np.array([j for i in range(m) for j in neighbours(i)])
        row = np.repeat(np.arange(m), 4)
        DistanceCalculator._edge_data = (row, col)

    def _simple_dropoff_distances(self, dropoffs):
        """Simple step distances from all cells to the nearest dropoff."""
        distances_to_all_dropoffs = np.array([
            simple_distances(to_index(dropoff)) for dropoff in dropoffs
        ])
        return np.min(distances_to_all_dropoffs, axis=0)

    def threat_costs_func(self, ship, threat_costs):
        """Necessary to keep Schedule costs in sync."""
        return 20.0 * packing_fraction(ship) * threat_costs

    def _threat_edge_costs(self, enemy_threat):
        """Edge costs describing avoiding enemies (fleeing)."""
        _row, col = DistanceCalculator._edge_data
        return enemy_threat[col]

    def _return_edge_costs(self, dropoffs):
        """Edge costs describing turns necessary to return to a dropoff."""
        dropoff_distances = self.simple_dropoff_distances
        row, col = DistanceCalculator._edge_data
        return 0.5 * (dropoff_distances[col] - dropoff_distances[row] + 1.0)

    def _traffic_edge_costs(self):
        """Edge costs describing avoiding or waiting for traffic."""
        m = game_map.height * game_map.width
        occupation = np.array([
            to_cell(j).is_occupied
            for i in range(m) for j in neighbours(i)
        ])
        return 0.8 * occupation

    def _movement_edge_costs(self, halite):
        """Edge costs describing basic movement."""
        return np.repeat(1.0 + halite / 750.0, 4)

    def _edge_costs(self, ship):
        """Edge costs for all edges in the graph.

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
            The term containing self.enemy_cost represents the fact that
            losing a ship to a collision costs a lot of turns.
        """
        movement_costs = self._movement_costs
        traffic_costs = self._traffic_costs
        return_costs = packing_fraction(ship) * self._return_costs
        threat_costs = self.threat_costs_func(ship, self._threat_costs)
        return movement_costs + traffic_costs + return_costs + threat_costs

    def _nearby_edges(self, ship, edge_costs, row, col):
        """Drop far away edges to reduce computation time."""
        subgraph_indices = np.array(list(neighbourhood(to_index(ship), 15)))
        edge_indices = np.concatenate((
            4 * subgraph_indices,
            4 * subgraph_indices + 1,
            4 * subgraph_indices + 2,
            4 * subgraph_indices + 3,
        ))
        return edge_costs[edge_indices], row[edge_indices], col[edge_indices]

    def _graph(self, ship):
        """Create a sparse matrix representing the game map graph."""
        m = game_map.height * game_map.width
        edge_costs = self._edge_costs(ship)
        row, col = DistanceCalculator._edge_data
        edge_costs, row, col = self._nearby_edges(ship, edge_costs, row, col)
        return csr_matrix((edge_costs, (row, col)), shape=(m, m))

    def _unreachable(self, dist_matrix, indices, target_index):
        """Set simple distances for unreachable cells (in the graph)."""
        for i, index in enumerate(indices):
            distance = 10.0 + simple_distance(index, target_index)
            dist_matrix[i][target_index] = distance

    def _postprocess(self, dist_matrix, indices):
        """Do some postprocessing on the result from dijkstra()."""
        for dropoff in self.dropoffs:
            dropoff_index = to_index(dropoff)
            if dist_matrix[0][dropoff_index] == np.inf:
                self._unreachable(dist_matrix, indices, dropoff_index)
        dist_matrix[dist_matrix == np.inf] = 99999.9

    def _indices(self, ship):
        """Shortest paths for the ship cell and its direct neighbours."""
        ship_index = to_index(ship)
        return (ship_index, ) + neighbours(ship_index)

    def _ship_shortest_path(self, ship):
        """Calculate shortest path costs to all cells."""
        graph = self._graph(ship)
        indices = self._indices(ship)
        dist_matrix = dijkstra(graph, indices=indices)
        self._postprocess(dist_matrix, indices)
        return dist_matrix, indices

    def _shortest_path(self):
        """Calculate shortest path costs for all ships."""
        dist_tuples = {}
        for ship in game.me.get_ships():
            dist_tuples[ship.id] = self._ship_shortest_path(ship)
        return dist_tuples

    def get_distance_from_index(self, ship, from_index, to_index):
        """Get the distance from index (near ship) to index."""
        dist_matrix, indices = self._dist_tuples[ship.id]
        return dist_matrix[indices.index(from_index)][to_index]

    def get_distances(self, ship):
        """Get an array of perturbed distances to all cells."""
        dist_matrix, _indices = self._dist_tuples[ship.id]
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


##############################################################################
#
# Interaction with enemies
#
##############################################################################


def enemy_ships():
    """Generator for all enemy ships."""
    return (
        enemy_ship
        for player in game.players.values() if player is not game.me
        for enemy_ship in player.get_ships()
    )


def enemy_threat():
    """Assign a value to every cell describing enemy threat."""
    m = game_map.height * game_map.width
    threat = np.zeros(m)
    for ship in enemy_ships():
        ship_index = to_index(ship)
        threat[ship_index] += 1.0 - packing_fraction(ship)
        for index in neighbours(ship_index):
            threat[index] += 1.0 - packing_fraction(ship)
    return threat


def _simple_ship_threat(ship):
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


def _bonus_neighbourhood(ship):
    """Generator for the indices of the bonus neighbourhood of a ship."""
    return neighbourhood(to_index(ship), 4)


def _index_count(index_func):
    """Loops over enemy ships and counts indices returned by index_func."""
    counted = Counter(
        index
        for ship in enemy_ships()
        for index in index_func(ship)
    )
    m = game_map.height * game_map.width
    index_count = np.zeros(m)
    for index, counted_number in counted.items():
        index_count[index] = counted_number
    return index_count


def enemies_in_bonus_range():
    """Calculate the number of enemies within bonus range for all cells."""
    return _index_count(_bonus_neighbourhood)


def global_threat():
    """Calculate enemy threat factor for all cells."""
    threat = _index_count(_simple_ship_threat)
    return 3.0 / (threat + 3.0)


def _nearby_enemy_ships(ship):
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
        if cell.is_occupied and cell.ship.owner != game.me.id
    ]


def _mining_probability(halite, ship):
    """Estimate the probability that a ship will mine the next turn."""
    if not can_move(ship):
        return 1.0
    ship_index = to_index(ship)
    simple_cost = halite / (simple_distances(ship_index) + 1.0)
    mining_cost = simple_cost[ship_index]
    moving_cost = np.delete(simple_cost, ship_index).max()
    cargo_factor = min(1.0, 10.0 * (1.0 - packing_fraction(ship)))
    return cargo_factor * mining_cost / (mining_cost + moving_cost)


def nearby_loot(halite, ship):
    """Calculate enemy halite near a ship that can be stolen.

    Strategy:
        Take into account the amount of collisions with the enemy player:
        - Keep track of how and to whom you lost your own ships.
        - Flee/attack more aggresively for aggresive players (tit-for-tat).
    """
    m = game_map.height * game_map.width
    loot = np.zeros(m)
    for enemy_ship in _nearby_enemy_ships(ship):
        dhalite = enemy_ship.halite_amount - ship.halite_amount
        if dhalite > 0:
            mining_probability = _mining_probability(halite, enemy_ship)
            enemy_index = to_index(enemy_ship)
            loot[enemy_index] += dhalite * mining_probability
            for index in neighbours(enemy_index):
                loot[index] += dhalite * (1 - mining_probability)
    return loot


##############################################################################
#
# MapData, the main class
#
##############################################################################


class MapData:
    """Analyzes the gamemap and provides useful data/statistics."""

    def __init__(self, _game):
        global game, game_map
        game = _game
        game_map = game.game_map
        self.halite = self._halite()
        self.dropoffs = [game.me.shipyard] + game.me.get_dropoffs()
        self.enemy_threat = enemy_threat()
        self.in_bonus_range = enemies_in_bonus_range()
        self.global_threat = global_threat()
        self.calculator = DistanceCalculator(self.halite, self.dropoffs, self.enemy_threat)
        self.halite_density = self._halite_density()
        self.ship_density = self._ship_density()

    def _halite(self):
        """Get an array of available halite on the map."""
        m = game_map.height * game_map.width
        return np.array([to_cell(i).halite_amount for i in range(m)])

    def _halite_density(self):
        """Get density of halite map with radius"""
        halite = self.halite.reshape(game_map.height, game_map.width)
        return calc_density(radius=15, array=halite).ravel()

    def _ship_density(self):
        """Get density of friendly - hostile ships"""
        radius = 9
        friendly = np.zeros(self.halite.shape)
        friendly_indices = [to_index(ship) for ship in game.me.get_ships()]
        friendly[friendly_indices] = 1
        friendly_density = calc_density(radius, friendly.reshape(game_map.height, game_map.width), count_self=False)
        hostile = np.zeros(self.halite.shape)
        hostile_indices = [to_index(ship)
                           for player in game.players.values() if player is not game.me
                           for ship in player.get_ships()]
        hostile[hostile_indices] = 1
        hostile_density = calc_density(radius=radius, array=hostile.reshape(game_map.height, game_map.width))
        return (friendly_density - hostile_density).ravel()

    def distance_dropoffs(self, ships):
        """Get a list of distances to the nearest dropoff for all ships."""
        dropoff_dists = self.calculator.simple_dropoff_distances
        return np.array([dropoff_dists[to_index(ship)] for ship in ships])

    def get_closest_dropoff(self, ship):
        """Get the dropoff that is closest to the ship."""
        return self.calculator.get_closest(ship, self.dropoffs)

    def free_turns(self, ship):
        """Get the number of turns that the ship can move freely."""
        dropoff = self.get_closest_dropoff(ship)
        distance = self.calculator.get_entity_distance(ship, dropoff)
        turns_left = constants.MAX_TURNS - game.turn_number
        return turns_left - math.ceil(distance)

    def get_distances(self, ship):
        """Get an array of perturbed distances to all cells."""
        return self.calculator.get_distances(ship)

    def get_distance(self, ship, index):
        """Get the perturbed distance from a ship an index (a cell)."""
        return self.calculator.get_distance(ship, index)

    def loot(self, ship):
        """Calculate enemy halite near a ship that can be stolen."""
        return nearby_loot(self.halite, ship)
