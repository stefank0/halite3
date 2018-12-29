#!/usr/bin/env python3
import json, logging, time
from statistics import median

from hlt import constants, Game
from scheduler import Scheduler
from mapdata import MapData, DistanceCalculator


CALIBRATION = False


if CALIBRATION:
    import argparse, yaml

    def get_parser():
        """Get argumentparser and add arguments"""
        parser = argparse.ArgumentParser('read *pars.yaml and use parameters in MyBot')
        parser.add_argument('inputfile', type=str, help='YAML input file containing keyword arguments')
        return parser


def create_schedule():
    """Creates a schedule based on the current game map."""
    map_data = MapData(game, Scheduler.ghost)
    scheduler = Scheduler(game, map_data)
    return scheduler.get_schedule()


def add_move_commands(command_queue):
    """Add movement commands to the command queue."""
    schedule = create_schedule()
    command_queue.extend(schedule.to_commands())


def other_players():
    """Get a list of the other players."""
    return [player for player in game.players.values() if player is not me]


def number_of_ships(player):
    """Get the number of ships of a player."""
    return len(player.get_ships())


def _ship_number_falling_behind():
    """Return True if our ship number isn't high compared to the others."""
    ship_numbers = [number_of_ships(player) for player in other_players()]
    return number_of_ships(me) <= median(ship_numbers)


def want_to_spawn():
    """Return True if we would like to spawn a new ship."""
    is_early_game = game.turn_number <= parameters['earlygame']
    is_late_game = game.turn_number >= (constants.MAX_TURNS - parameters['endgame'])
    is_mid_game = (not is_early_game) and (not is_late_game)
    return is_early_game or (is_mid_game and _ship_number_falling_behind())


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
game.ready("Schildpad")

# Define some globals for convenience.
me = game.me
game_map = game.game_map


if CALIBRATION:
    args = get_parser().parse_args()
    with open(args.inputfile) as y:
        parameters = yaml.load(y)
        parameters['inputfile'] = args.inputfile
else:
    number_of_players = len(game.players)
    map_size = game.game_map.height
    try:
        with open('parameters.json') as f:
            all_parameters = json.load(f)
    except IOError:
        with open('../parameters.json') as f:
            all_parameters = json.load(f)
    parameters = all_parameters[str(number_of_players)][str(map_size)]


# Play the game.
while True:
    game.update_frame()
    start = time.time()
    command_queue = generate_commands()
    time_taken = time.time() - start
    #logging.info(time_taken)
    if time_taken > 1.4:
        #log_profiling()
        DistanceCalculator.reduce_radius()
    elif time_taken < 0.9:
        DistanceCalculator.increase_radius()

    game.end_turn(command_queue)
