#!/usr/bin/env python3
import logging, time
bot_start = time.time()
import numpy as np

from hlt import constants, Game
from scheduler import Scheduler
from parameters import param, load_parameters
from mapdata import MapData, DistanceCalculator, to_cell, ship_number_falling_behind


def create_schedule():
    """Creates a schedule based on the current game map."""
    map_data = MapData(game, Scheduler.ghost)
    scheduler = Scheduler(game, map_data)
    return scheduler.get_schedule()


def add_move_commands(command_queue):
    """Add movement commands to the command queue."""
    schedule = create_schedule()
    command_queue.extend(schedule.to_commands())


def _new_ships_are_all_mine():
    """Return True if the last spawned ships are ours."""
    ship_ids = [ship.id for player in game.players.values() for ship in player.get_ships()]
    if len(ship_ids) > 5 and all([me.has_ship(ship_id) for ship_id in sorted(ship_ids)[-5:]]):
        return True
    return False


def old_want_to_spawn():
    """Old implementation of want_to_spawn()."""
    is_late_game = game.turn_number > 0.75 * constants.MAX_TURNS
    if _new_ships_are_all_mine() or is_late_game:
        return False
    return ship_number_falling_behind()


def new_want_to_spawn():
    """New implementation of want_to_spawn()."""
    turnremain = constants.MAX_TURNS - game.turn_number
    m = game_map.height * game_map.width
    halite = np.array([to_cell(i).halite_amount for i in range(m)])
    haliteremain = halite.sum()
    return (param['spawn_intercept'] +
            param['spawn_turnremain'] * turnremain +
            param['spawn_haliteremain'] * haliteremain) > 2000


def want_to_spawn():
    """Return True if we would like to spawn a new ship."""
    return new_want_to_spawn() or old_want_to_spawn()


def can_spawn(command_queue):
    """Return True if it is possible to spawn a new ship."""
    halite = Scheduler.free_halite
    enough_halite = halite >= constants.SHIP_COST
    shipyard_free = not game_map[me.shipyard].is_occupied
    return enough_halite and shipyard_free


def add_spawn_command(command_queue):
    """If possible and desirable, add the spawn command."""
    if can_spawn(command_queue) and want_to_spawn():
        command_queue.append(me.shipyard.spawn())


def unmark_shipyard():
    """Unmark shipyard, so that can_spawn() works properly."""
    game_map[me.shipyard].ship = None


def generate_commands():
    """Generate a list of commands based on the current game state."""
    unmark_shipyard()
    command_queue = []
    add_move_commands(command_queue)
    add_spawn_command(command_queue)
    return command_queue


def log_profiling():
    """Profiling to get insight in performance issues."""
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    generate_commands()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    logging.info(s.getvalue())
    s.close()


# Initialize the game.
game = Game()
load_parameters(game)
logging.info(param)

# Define some globals for convenience.
me = game.me
game_map = game.game_map

MapData(game, None)
while(time.time() - bot_start < 9.7 and DistanceCalculator.needs_precompute()):
    DistanceCalculator.precompute()

# Play the game.
game.ready("TeamSchildpad")
while True:
    game.update_frame()
    start = time.time()
    command_queue = generate_commands()
    # time_taken = time.time() - start
    # logging.info(time_taken)
    # if time_taken > 1.4:
        # log_profiling()

    while(time.time() - start < 1.7 and DistanceCalculator.needs_precompute()):
        DistanceCalculator.precompute()
    game.end_turn(command_queue)
