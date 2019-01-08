#!/usr/bin/env python3
import logging, time
from statistics import median

from hlt import constants, Game
from scheduler import Scheduler
from parameters import param, load_parameters
from mapdata import MapData, DistanceCalculator, other_players


def create_schedule():
    """Creates a schedule based on the current game map."""
    scheduler = Scheduler(game, map_data)
    return scheduler.get_schedule()


def add_move_commands(command_queue):
    """Add movement commands to the command queue."""
    schedule = create_schedule()
    command_queue.extend(schedule.to_commands())


def number_of_ships(player):
    """Get the number of ships of a player."""
    return len(player.get_ships())


def _ship_number_falling_behind():
    """Return True if our ship number isn't high compared to the others."""
    ship_numbers = [number_of_ships(player) for player in other_players()]
    return number_of_ships(me) <= median(ship_numbers)


def _new_ships_are_all_mine():
    """Return True if the last spawned ships are ours."""
    ship_ids = [ship.id for player in game.players.values() for ship in player.get_ships()]
    if len(ship_ids) > 5 and all([me.has_ship(ship_id) for ship_id in sorted(ship_ids)[-5:]]):
        return True
    return False


def want_to_spawn():
    """Return True if we would like to spawn a new ship."""
    turnremain = constants.MAX_TURNS - game.turn_number
    haliteremain = map_data.halite.sum()
    return (param['spawn_intercept'] +
            param['spawn_turnremain'] * turnremain +
            param['spawn_haliteremain'] * haliteremain) > 750


# def want_to_spawn():
#     """Return True if we would like to spawn a new ship."""
#     if _new_ships_are_all_mine():
#         return False
#     is_early_game = game.turn_number <= param['earlygame']
#     is_late_game = game.turn_number >= (constants.MAX_TURNS - param['endgame'])
#     is_mid_game = (not is_early_game) and (not is_late_game)
#     return is_early_game or (is_mid_game and _ship_number_falling_behind())


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
game.ready("Schildpad")

# Define some globals for convenience.
me = game.me
game_map = game.game_map

# Play the game.
while True:
    game.update_frame()
    start = time.time()
    map_data = MapData(game, Scheduler.ghost)
    command_queue = generate_commands()
    time_taken = time.time() - start
    # logging.info(time_taken)
    if time_taken > 1.4:
        # log_profiling()
        DistanceCalculator.reduce_radius()
    elif time_taken < 0.9:
        DistanceCalculator.increase_radius()

    game.end_turn(command_queue)
