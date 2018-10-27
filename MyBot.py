#!/usr/bin/env python3

import logging, hlt, time
from hlt import constants, Direction, Position
from statistics import median
from scheduling import Schedule

#
# Idee: definieer hoeveel stappen halite waard is, om te bepalen of er een dropoff moet komen en zo ja waar. Als
# besloten waar, dan moet deze al beschikbaar zijn als toekomstige dropoff, zodat ships daar alvast naartoe kunnen
# bewegen (de eerste die er is, maakt de dropoff aan).
#

returning_to_shipyard = set()


def get_checked_positions(distance, ship):
    positions = []
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            if abs(i) + abs(j) > distance:
                continue
            position = ship.position + Position(i, j)
            positions.append(position)
    return positions


def find_new_halite(ship):
    halite_amount = 0
    distance = 0
    while halite_amount < 0.05 * constants.MAX_HALITE and distance < 10:
        distance += 1
        checked_positions = get_checked_positions(distance, ship)
        destination = max(checked_positions, key=lambda position: game_map[position].halite_amount)
        halite_amount = game_map[destination].halite_amount
    return destination


def returning(ship):
    return (ship.halite_amount > 0.75 * constants.MAX_HALITE) or (ship.id in returning_to_shipyard)


def mining(ship, local_halite):
    return (local_halite > 0.05 * constants.MAX_HALITE) or (0.2 * local_halite > ship.halite_amount)


def create_schedule():
    # Preprocessing.
    for ship in me.get_ships():
        if ship.halite_amount < 0.25 * constants.MAX_HALITE:
            returning_to_shipyard.discard(ship.id)

    # Move ships.
    schedule = Schedule(game_map)
    for ship in me.get_ships():
        local_halite = game_map[ship].halite_amount
        if returning(ship):
            returning_to_shipyard.add(ship.id)
            destination = me.shipyard.position
        elif mining(ship, local_halite):
            destination = ship.position
        else:
            destination = find_new_halite(ship)
        schedule.assign(ship, destination)
    return schedule


def add_move_commands(command_queue):
    """Add movement commands to the command queue."""
    start = time.time()
    schedule = create_schedule()
    end = time.time()
    logging.info(end - start)
    start = time.time()
    command_queue.extend(schedule.to_commands())
    end = time.time()
    logging.info(end - start)


def other_players():
    """Get a list of the other players."""
    return [player for player in game.players.values() if not player is me]


def number_of_ships(player):
    """Get the number of ships of a player."""
    return len(player.get_ships())


def _ship_number_falling_behind():
    """Return True if our ship number isn't high compared to the others."""
    ship_numbers = [number_of_ships(player) for player in other_players()]
    return number_of_ships(me) <= median(ship_numbers)


def want_to_spawn():
    """Return True if we would like to spawn a new ship."""
    is_early_game = game.turn_number <= constants.MAX_TURNS / 2
    is_late_game = game.turn_number >= constants.MAX_TURNS * 3 / 4
    is_mid_game = (not is_early_game) and (not is_late_game)
    return is_early_game or (is_mid_game and _ship_number_falling_behind())


def can_spawn():
    """Return True if it is possible to spawn a new ship."""
    enough_halite = me.halite_amount >= constants.SHIP_COST
    shipyard_free = not game_map[me.shipyard].is_occupied
    return enough_halite and shipyard_free


def add_spawn_command(command_queue):
    """If possible and desirable, add the spawn command."""
    if can_spawn() and want_to_spawn():
        command_queue.append(me.shipyard.spawn())


def generate_commands():
    """Generate a list of commands based on the current game state."""
    command_queue = []
    add_move_commands(command_queue)
    add_spawn_command(command_queue)
    return command_queue


def mark_safe():
    """Undo marking that was done in game.update_frame()."""
    for y in range(game_map.height):
        for x in range(game_map.width):
            game_map[Position(x, y)].ship = None


# Initialize the game.
game = hlt.Game()
game.ready("Schildpad")

# Define some globals for convenience.
me = game.me
game_map = game.game_map

# Play the game.
while True:
    game.update_frame()
    mark_safe()
    command_queue = generate_commands()
    game.end_turn(command_queue)
