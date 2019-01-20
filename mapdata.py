import logging, math, time
from statistics import median
from collections import Counter

from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np

from hlt import Position, constants
from hlt.entity import Shipyard
from parameters import param


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
    return game_map._cells[y][x]


def can_move(ship):
    """True if a ship is able to move."""
    necessary_halite = math.floor(0.1 * game_map[ship].halite_amount)
    return necessary_halite <= ship.halite_amount


def packing_fraction(ship):
    """Get the packing/fill fraction of the ship."""
    return ship.halite_amount / constants.MAX_HALITE


def target(origin, direction):
    """Calculate the target cell if the ship moves in the given direction."""
    return game_map[origin.directional_offset(direction)]


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


def circle(index, radius):
    """Get indices at a circle around an index."""
    h = game_map.height
    w = game_map.width
    x = index % w
    y = index // w
    return [
        ((x + dx) % w) + (w * ((y + dy) % h))
        for dx in range(-radius, radius + 1)
        for dy in {-radius + abs(dx), radius - abs(dx)}
    ]


def density(base_density, radius):
    """Smooth/distribute a base density over a region."""
    base_density_sum = base_density.sum()
    if base_density_sum == 0.0:
        return base_density
    base_density = base_density.reshape(game_map.height, game_map.width)
    density = np.zeros((game_map.height, game_map.width))
    for dx in range(-radius, radius + 1):
        for dy in range(-radius + abs(dx), radius + 1 - abs(dx)):
            factor = 1.0 - (abs(dx) + abs(dy)) / (radius + 1.0)
            density += factor * np.roll(base_density, (dx, dy), (0, 1))
    density = density.ravel()
    return density * (base_density_sum / density.sum())


def ship_density(ships, radius):
    """Transform a list of ships into a density on the game map."""
    base_density = np.zeros(game_map.height * game_map.width)
    ship_indices = [to_index(ship) for ship in ships]
    base_density[ship_indices] = 1.0
    return density(base_density, radius)


def nearby_ships(ship, ships, radius):
    """Return a list of nearby ships out of ships."""
    return [
        other_ship
        for other_ship in ships
        if simple_distance(to_index(ship), to_index(other_ship)) <= radius
    ]


class LinearSum:
    """Wrapper for linear_sum_assignment() from scipy to avoid timeouts."""

    _time_saving_mode1 = False
    _time_saving_mode2 = False

    @classmethod
    def simple_assignment(cls, cost_matrix):
        """Simple heuristic/non-optimal/greedy assignment."""
        if cost_matrix.size == 0:
            return [], []
        cost = cost_matrix - cost_matrix.max()
        row_ind = []
        col_ind = []
        while True:
            row, col = np.unravel_index(cost.argmin(), cost.shape)
            if cost[row, col] < 0.0:
                row_ind.append(row)
                col_ind.append(col)
                cost[row, :] = 0.0
                cost[:, col] = 0.0
            else:
                break
        for row in range(cost_matrix.shape[0]):
            if not row in row_ind:
                row_ind.append(row)
                col = cost_matrix[row, :].argmin()
                col_ind.append(col)
        return row_ind, col_ind

    @classmethod
    def _add_to_cluster(cls, cluster, ship, ships, radius=2):
        """Add ship to cluster and search other ships for the cluster."""
        cluster.append(ship)
        for other_ship in nearby_ships(ship, ships, radius):
            if other_ship not in cluster:
                cls._add_to_cluster(cluster, other_ship, ships, radius)

    @classmethod
    def _already_in_cluster(cls, clusters, ship):
        """Test if the ship is already in another cluster."""
        for cluster in clusters:
            if ship in cluster:
                return True
        return False

    @classmethod
    def _get_clusters(cls, ships, cluster_mode):
        """Create the ship clusters."""
        clusters = []
        for ship in ships:
            if cls._already_in_cluster(clusters, ship):
                continue
            cluster = []
            cls._add_to_cluster(cluster, ship, ships)
            clusters.append(cluster)
        return clusters

    @classmethod
    def _efficient_assignment(cls, cost_matrix, ships, cluster_mode):
        """Cluster ships and solve multiple linear sum assigments.

        Note:
            The Hungarian algorithm has complexity n^3, so it is much more
            efficient to solve several small problems than it is to solve one
            large problem. The ships are split into groups in such a way that
            the assignment in Schedule has exactly the same result.
        """
        clusters = cls._get_clusters(ships, cluster_mode)
        row_inds = []
        col_inds = []
        for cluster in clusters:
            indices = np.array([ships.index(ship) for ship in cluster])
            partial_cost_matrix = cost_matrix[indices, :]
            if len(cluster) > 50 and game_map.height == 64 and len(game.players) == 4:
                row_ind, col_ind = cls.simple_assignment(partial_cost_matrix)
            else:
                row_ind, col_ind = linear_sum_assignment(partial_cost_matrix)
            row_inds += [int(x) for x in indices[row_ind]]
            col_inds += [int(x) for x in col_ind]
        return row_inds, col_inds

    @classmethod
    def assignment(cls, cost_matrix, ships, cluster_mode=False):
        """Wraps linear_sum_assignment()."""
        if cluster_mode or cls._time_saving_mode2:
            return cls._efficient_assignment(cost_matrix, ships, cluster_mode)
        elif cls._time_saving_mode1:
            start = time.time()
            row_ind, col_ind = cls.simple_assignment(cost_matrix)
            stop = time.time()
            if stop - start > 0.25:
                cls._time_saving_mode2 = True
                logging.info("Switching to time saving mode 2.")
            return row_ind, col_ind
        else:
            start = time.time()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            stop = time.time()
            if stop - start > 0.25:
                cls._time_saving_mode1 = True
                logging.info("Switching to time saving mode 1.")
            return row_ind, col_ind


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


def simple_distances(index, indices):
    """Get an array of the actual step distances to specific cells."""
    height = game_map.height
    width = game_map.width
    dx = np.abs(indices % width - index % width)
    dy = np.abs(indices // width - index // width)
    return np.minimum(dx, width - dx) + np.minimum(dy, height - dy)


_all_simple_distance_cache = {}

def all_simple_distances(index):
    """Get an array of the actual step distances to all cells."""
    if index in _all_simple_distance_cache:
        return _all_simple_distance_cache[index]
    else:
        indices = np.arange(game_map.width * game_map.height)
        distances = simple_distances(index, indices)
        _all_simple_distance_cache[index] = distances
        return distances


class DistanceCalculator:
    """Calculates shortest path distances for all ships."""

    _edge_data = None
    _dijkstra_radius = 15
    _expand_array_cache = {}
    _next_precompute = 0

    @classmethod
    def _initialize_edge_data(cls):
        """Store edge_data for create_graph() on the class for performance."""
        m = game_map.height * game_map.width
        col = np.array([j for i in range(m) for j in neighbours(i)])
        row = np.repeat(np.arange(m), 4)
        cls._edge_data = (row, col)

    @classmethod
    def needs_precompute(cls):
        """Test if precomputation is finished."""
        return cls._next_precompute < game_map.height * game_map.width

    @classmethod
    def precompute(cls):
        """Fill the _expand_array_cache."""
        cls._expand_arrays(cls._next_precompute)
        cls._next_precompute += 1

    @classmethod
    def _compute_expand_arrays(cls, index):
        """Compute arrays necessary to perform expansion."""
        radius = cls._dijkstra_radius + 1
        indices = np.flatnonzero(all_simple_distances(index) > radius)
        boundary = np.array(circle(index, radius))
        distances = np.array([simple_distances(i, indices) for i in boundary])
        closest = boundary[distances.argmin(0)]
        distance = 2.0 * distances.min(0)
        return indices, closest, distance

    @classmethod
    def _expand_arrays(cls, ship_index):
        """Arrays necessary to perform the expansion."""
        if ship_index in cls._expand_array_cache:
            return cls._expand_array_cache[ship_index]
        arrays = cls._compute_expand_arrays(ship_index)
        cls._expand_array_cache[ship_index] = arrays
        return arrays

    def __init__(self, dropoffs, halite):
        if self._edge_data is None:
            self._initialize_edge_data()
        self.collision_area = self._collision_area()
        self.dropoffs = dropoffs
        self.troll_indices = self._troll_indices()
        self.simple_dropoff_distances = self._simple_dropoff_distances(dropoffs)
        self.enemy_dropoff_distances = self._enemy_dropoff_distances()
        self._traffic_costs = self._traffic_edge_costs()
        self._movement_costs = self._movement_edge_costs(halite)
        self.threat_factor = self._threat_factor()
        self._dist_tuples = self._shortest_path()
        self.ghost_distances = self._ghost_distances()

    def _simple_dropoff_distances(self, dropoffs):
        """Simple step distances from all cells to the nearest dropoff."""
        all_dropoff_distances = np.array([
            all_simple_distances(to_index(dropoff))
            for dropoff in dropoffs
        ])
        return np.min(all_dropoff_distances, axis=0)

    def _enemy_dropoff_distances(self):
        """Step distances from all cells to the nearest enemy dropoff."""
        dropoffs = list(enemy_dropoffs()) + list(enemy_shipyards())
        return self._simple_dropoff_distances(dropoffs)

    def _ghost_distances(self):
        """Calculate distances used in GhostDropoff, uses the second ship."""
        distances = [self.get_distances(ship) for ship in game.me.get_ships()]
        return self._second_distances(distances)

    def _collision_area(self):
        """Determine area in which collisions are OK."""
        my_ships = game.me.get_ships()
        ships_with_space = (s for s in my_ships if s.halite_amount < 500)
        second_distances = self.second_ship_distances(ships_with_space)
        second_enemy_distances = self.second_ship_distances(enemy_ships())
        return second_enemy_distances > second_distances

    def second_ship_distances(self, ships):
        """Calculate the distance of the second closest ship for all cells."""
        distances = [all_simple_distances(to_index(ship)) for ship in ships]
        return self._second_distances(distances)

    def _second_distances(self, distances):
        """Return the second closests distance values from distances."""
        if len(distances) <= 1:
            return np.full(game_map.height * game_map.width, 999.9)
        return np.partition(distances, 1, 0)[1]

    def _troll_indices(self):
        """Indices that could be occupied by enemy trolls."""
        dropoffs = [game.me.shipyard] + game.me.get_dropoffs()
        dropoff_indices = [to_index(dropoff) for dropoff in dropoffs]
        near_dropoff_indices = [
            index
            for dropoff_index in dropoff_indices
            for index in neighbours(dropoff_index)
        ]
        return dropoff_indices + near_dropoff_indices

    def threat_to_self(self, ship):
        """Cost representing threat to current position."""
        index = to_index(ship)
        cells = (to_cell(i) for i in neighbours(index))
        enemy_ships = [c.ship for c in cells if c.is_occupied and c.ship.owner != game.me.id]
        if enemy_ships:
            d = max(ship.halite_amount - s.halite_amount for s in enemy_ships)
            if d > param['self_threat_threshold']:
                return self.threat_factor[index] * 2.0 ** (d / 75.0)
        return 0.0

    def _threat_factor(self):
        """Factor common to all threat_edge_costs()."""
        is_4player = len(game.players) == 4
        is_endgame = game.turn_number > 0.75 * constants.MAX_TURNS
        factor = param['threat'] * (10.0 - 9.0 * self.collision_area)
        if (is_4player and not is_endgame) or ship_number_falling_behind():
            return 10.0 * factor
        return factor

    def _threat_edge_costs(self, ship):
        """Edge costs describing avoiding enemies (fleeing)."""
        threat = np.zeros(game_map.height * game_map.width)
        index = to_index(ship)
        for enemy_ship in enemy_ships():
            enemy_index = to_index(enemy_ship)
            if (simple_distance(index, enemy_index) > self._dijkstra_radius or
                enemy_index in self.troll_indices):
                continue
            d = ship.halite_amount - enemy_ship.halite_amount
            threat_value = 2.0 ** (d / 75.0)
            threat[enemy_index] += 4.0 * threat_value + 3.0
            if can_move(enemy_ship):
                for i in neighbours(enemy_index):
                    threat[i] += threat_value
        threat *= self.threat_factor
        _row, col = self._edge_data
        return threat[col]

    def _traffic_edge_costs(self):
        """Edge costs describing avoiding or waiting for traffic."""
        m = game_map.height * game_map.width
        occupation = np.array([
            to_cell(j).is_occupied
            for i in range(m) for j in neighbours(i)
        ])
        return min(0.99, param['traffic_factor']) * occupation

    def _movement_edge_costs(self, halite):
        """Edge costs describing basic movement.

        Note
            The edge cost is chosen such that the shortest path is mainly based
            on the number of steps necessary, but also slightly incorporates
            the halite costs of moving. Therefore, the most efficient path is
            chosen when there are several shortest distance paths.
            More solid justification: if mining yields 75 halite on average,
            one mining turn corresponds to moving over 75/(10%) = 750 halite.
            Therefore, moving over 1 halite corresponds to 1/750 of a turn.
        """
        halite_cost = np.floor(0.1 * halite)
        return np.repeat(1.0 + halite_cost / param['mean_halite'], 4)

    def _edge_costs(self, ship):
        """Edge costs for all edges in the graph."""
        threat_costs = self._threat_edge_costs(ship)
        return self._movement_costs + self._traffic_costs + threat_costs

    def _nearby_edges(self, ship, edge_costs, row, col):
        """Keep only nearby edges to reduce computation time."""
        radius = self._dijkstra_radius
        ship_neighbourhood = neighbourhood(to_index(ship), radius)
        subgraph_indices = np.array(list(ship_neighbourhood))
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
        row, col = self._edge_data
        if game_map.width > 40:
            edge_costs, row, col = self._nearby_edges(ship, edge_costs, row, col)
        return csr_matrix((edge_costs, (row, col)), shape=(m, m))

    def _expand(self, dist_matrix, ship_index):
        """Expand the region for which distances are set in dist_matrix."""
        indices, closest, distance = self._expand_arrays(ship_index)
        for i in range(5):
            dist_matrix[i, indices] = dist_matrix[i, closest] + distance

    def _indices(self, ship):
        """Shortest paths for the ship cell and its direct neighbours."""
        ship_index = to_index(ship)
        return (ship_index,) + neighbours(ship_index)

    def _ship_shortest_path(self, ship):
        """Calculate shortest path costs to all cells."""
        graph = self._graph(ship)
        indices = self._indices(ship)
        dist_matrix = dijkstra(graph, indices=indices)
        self._expand(dist_matrix, indices[0])
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


def other_players():
    """Generator for all other players."""
    return (player for player in game.players.values() if player is not game.me)


def enemy_ships():
    """Generator for all enemy ships."""
    return (ship for player in other_players() for ship in player.get_ships())


def enemy_dropoffs():
    """Generator for all enemy dropoffs."""
    return (dropoff for player in other_players() for dropoff in player.get_dropoffs())


def enemy_shipyards():
    """Generator for all enemy shipyards."""
    return (player.shipyard for player in other_players())


def number_of_ships(player):
    """Get the number of ships of a player."""
    return len(player.get_ships())


def ship_number_falling_behind():
    """Return True if our ship number isn't high compared to the others."""
    ship_numbers = [number_of_ships(player) for player in other_players()]
    is_early_game = game.turn_number < 0.5 * constants.MAX_TURNS
    threshold = median(ship_numbers) if is_early_game else min(ship_numbers)
    return number_of_ships(game.me) <= threshold


def _bonus_neighbourhood(ship):
    """Generator for the indices of the bonus neighbourhood of a ship."""
    return neighbourhood(to_index(ship), 4)


def enemies_in_bonus_range():
    """Calculate the number of enemies within bonus range for all cells."""
    counted = Counter(
        index
        for ship in enemy_ships()
        for index in _bonus_neighbourhood(ship)
    )
    in_bonus_range = np.zeros(game_map.height * game_map.width)
    for index, counted_number in counted.items():
        in_bonus_range[index] = counted_number
    return in_bonus_range


##############################################################################
#
# MapData, the main class
#
##############################################################################


class MapData:
    """Analyzes the gamemap and provides useful data/statistics."""

    def __init__(self, _game, ghost_dropoff):
        global game, game_map
        game = _game
        game_map = game.game_map
        self.turn_number = game.turn_number
        self.halite = self._halite()
        self.dropoffs = [game.me.shipyard] + game.me.get_dropoffs()
        self.in_bonus_range = enemies_in_bonus_range()
        self.all_dropoffs = self.dropoffs + [ghost_dropoff] if ghost_dropoff else self.dropoffs
        self.calculator = DistanceCalculator(self.all_dropoffs, self.halite)
        self.halite_density = self._halite_density()
        self.density_difference = self._ship_density_difference()
        hostile_density3 = ship_density(enemy_ships(), 3)
        friendly_density3 = ship_density(game.me.get_ships(), 3)
        self.density_difference3 = friendly_density3 - hostile_density3
        self.base_loot = self._base_loot()

    def _halite(self):
        """Get an array of available halite on the map."""
        m = game_map.height * game_map.width
        halite = np.array([to_cell(i).halite_amount for i in range(m)])
        for i in range(m):
            cell = to_cell(i)
            if cell.is_occupied and cell.ship.owner != game.me.id:
                # Halite is already gathered by enemy.
                halite[i] = max(halite[i] - param['halite_subtract'], 0)
        return halite

    def _halite_density(self):
        """Get density of halite map with radius"""
        return density(self.halite, 10)

    def _ship_density(self, ships, radius):
        """Get density of ships."""
        ship_density = np.zeros(game_map.height * game_map.width)
        ship_indices = [to_index(ship) for ship in ships]
        ship_density[ship_indices] = 1
        return density(ship_density, radius)

    def _ship_density_difference(self):
        """Get density of friendly - hostile ships"""
        friendly_density = self._ship_density(game.me.get_ships(), 8)
        self.hostile_density = self._ship_density(enemy_ships(), 8)
        return friendly_density - self.hostile_density

    def _perturbed_dropoff_distance(self, ship, dropoff):
        """Higher Shipyard distance to encourage moving to dropoffs."""
        is_shipyard = isinstance(dropoff, Shipyard)
        is_early = game.turn_number < 0.4 * constants.MAX_TURNS
        distance = self.get_entity_distance(ship, dropoff)
        return distance + 5 if (is_shipyard and is_early) else distance

    def get_closest_dropoff(self, ship):
        """Get the dropoff that is closest to the ship."""
        key = lambda dropoff: self._perturbed_dropoff_distance(ship, dropoff)
        return min(self.all_dropoffs, key=key)

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

    def get_entity_distance(self, ship, entity):
        """"Get the perturbed distance from a ship to an Entity."""
        return self.calculator.get_entity_distance(ship, entity)

    def _base_loot(self):
        """Define a base loot to be used in loot()."""
        base_loot = np.zeros(game_map.height * game_map.width)
        dropoff_dists = self.calculator.simple_dropoff_distances
        for enemy_ship in enemy_ships():
            enemy_index = to_index(enemy_ship)
            loot = enemy_ship.halite_amount
            if self.calculator.collision_area[enemy_index]:
                base_loot[enemy_index] = max(base_loot[enemy_index], loot)
                for index in neighbours(enemy_index):
                    if dropoff_dists[index] > dropoff_dists[enemy_index]:
                        base_loot[index] = max(base_loot[index], loot)
        return base_loot

    def loot(self, ship):
        """Calculate enemy halite near a ship that can be stolen.

        Strategy:
            Take into account the amount of collisions with the enemy player:
            - Keep track of how and to whom you lost your own ships.
            - Flee/attack more aggresively for aggresive players (tit-for-tat).
        """
        is_4player = len(game.players) == 4
        is_endgame = game.turn_number > 0.75 * constants.MAX_TURNS
        if is_4player and not is_endgame:
            return np.zeros(game_map.height * game_map.width)
        return param['lootfactor'] * (self.base_loot - ship.halite_amount)
