# TeamSchildpad

## Summary

### Moving/mining
The main steps of our bot are:
a) Create graphs representing the game map and calculate distances.
b) Assign ships to destinations using these distances.
c) Assign commands based on these destinations.

a) The game map is viewed as a graph, with edges connecting adjacent cells. The cost of moving over an edge contains costs accounting for traffic jams/enemy threat/fuel usage and are ship specific*. For these graphs (one graph per ship), distances are computed to all cells (using the Dijkstra algorithm only for a subgraph in order to stay within 2sec computation time).

\* An empty ship experiences less threat from enemy ships. The implementation is such that there is more or less a threshold separating scary from not scary.

b) All cells are scored for each ship: the score of cell x equals the maximum average halite gathered per turn out of a small set scenarios that involve moving to cell x (scenarios: mining up to 5 turns on cell x or moving/mining its best neighbor). Halite carried by enemy ships is also mixed in*, as are turns necessary to return to a dropoff (counted fully if the ship completely fills up to 1000, by ratio if the ship only partially fills). Based on these scores, ships are assigned to destinations, under the constraint that no destination is chosen twice (Hungarian algorithm, matching ships to map cells, maximizing the sum of the scores). If, for a certain ship, the expected total gathered halite in the rest of the game is very low, the ship will go on suicide mission defending our closest dropoff. Ships return to a dropoff when they are nearly full or when the average halite (the score) of the chosen destination is low compared (in some way) to the currently carried halite. Once a ship chooses to return to a dropoff, it is forced to return until it actually reaches a dropoff.

\* Actually the difference in carried halite compared to our ship, under some conditions, such as being in friendly territory. This difference is placed at 3 locations: the location of the enemy ship and the two adjacent locations furthest from our dropoffs, in order to cut off the escape route of the enemy ship and to lure in more than one of our ships to work together.

c) All moves are scored for each ship: the score of move x equals the distance of the shortest path to the destination c given move x (from a to b): score = edge cost from a to b + shortest path cost from b to c. Assign commands, minimizing the sum of these scores, under the constraint of zero self collisions (Hungarian algorithm again, matching ships to map cells). The result: if possible, all ships choose the shortest path, if not, the summed additional distance is minimized.

### Spawning
We tried to make an estimate of the return on investment of building a ship, based on real game data. We spawn if this estimate clearly indicates that spawning is a good idea or when our ship number is behind on the opponent ship numbers (early game: > median, mid game: > minimum, end game: don't spawn).

### Dropoffs
Attempts are made to plan a new dropoff when the ship-to-dropoff ratio reaches a certain point. A search area is determined, consisting of cells that meet certain criteria: close enough such that at least some ships will choose it as dropoff destination, yet not to close to existing dropoffs, high enough halite density and not controlled by the enemy. Basically, the cell with highest halite density is then chosen as spawn location for the new dropoff. If a suitable location is found, just enough halite is kept in reserve (taking into account halite carried by ships reaching a dropoff/shipyard before the new dropoff can be built). From that moment, the dropoff is taken into account as every other dropoff and the first ship getting there actually builds it. The only difference with regular dropoffs is that the planned dropoff can move a single step each turn, so that the dropoff itself can flee (a bit) from the enemy if necessary, and respond to changes in general.

### Python
Using Python to build the bot came with some challenges regarding performance. We needed to rely heavily on NumPy, which was OK. However, straightforward usage of the SciPy implementations of the Hungarian/Dijkstra algorithm resulted in timeouts, and the time was spent inside of the SciPy libraries. Because of that we needed to change the problem a bit / decrease the problem size (perturbing the problem/solution) and to carefully cache/precompute results that could be reused. Despite these performance problems, we would probably still use Python for our next bot, because we like using NumPy, there are enough options we did not explore (Cython, PyCUDA, ..) and in the end it worked out for this one.


## NOT TODO
- Performance improvements (linear_sum_assignment in Cython)(simpler/faster shortest path algorithm)(PyCUDA GPU).
- Improve attack and enemy_threat (Tit-for-tat, keep track of attacks on stationary ships as indication of aggressiveness)
- Predict what enemy ship will do (based on cell/ship halite, ships nearby, distance to dropoff, can_move)(use ML, possibly even predict against whom we are playing)(make a cost array like for our own ships to predict next move).
- Better manage tanking turns for ships with very low cargo.
- Make sure you never end up with slightly less than 1000 halite when you need 1000 to create a ship.
- Estimate investment return for dropoffs.
- Do more with ships that have nothing to lose: Troll/attack the enemy base. Protect full ships.
- Time dependent parameters (parameters such as those in threat_edge_costs, because losing a ship at the beginning of the game is much worse than at the end)(a+(b-a)(t/T) if t<T else b+(c-b)(t-T)/(Tmax-T))(instead of depending on time/turnnumber, let a parameter depend on available halite on the map, or something else)(find patterns in training data to determine what the parameters should depend on)
- Try to get bonus without giving the opponent a bonus (Encourage spreading of ships when near the enemy).
- Avoid going to high halite cells with ships that are nearly full (only mine cells that you can mine to the bottom).
- Calculate cost array for fake ship on dropoff, to be used in the return to dropoff decision (estimate average halite for even more turns, including dropping off halite at a dropoff).

# Halite III
General information from Halite III adjusted to our project.

## Halite III components
* /docs, contains API of the game of Halite 
* /hlt, contains modifiable helper functions for your bot
* /misc, contains command scripts and executables to run an example game (.bat for windows) (.sh for MacOS, Linux)
* /replays, contains replays and error files
* MyBot.py, TeamSchildpad bot
* scheduler.py, module to assign ships to destinations
* schedule.py, module to generate commands from assignments
* mapdata.py, module with useful general functions and MapData

## Testing your bot locally
* Run run_game.bat (Windows) and run_game.sh (MacOS, Linux) to run a game of Halite III. By default, these scripts run a game of your MyBot.py bot vs. itself.  You can modify the board size, map seed, and the opponents of test games using the CLI.

## CLI
The Halite executable comes with a command line interface (CLI). Run `$ ./halite --help` to see a full listing of available flags.


## Local viewer
* Fluorine can be used to view replays locally https://github.com/fohristiwhirl/fluorine/releases

## Submitting your bot
* Zip your MyBot.{extension} file and /hlt directory together.
* Submit your zipped file here: https://beta.halite.io/play-programming-challenge

## Compiling your bot on our game servers
* Your bot has `10 minutes` to install dependencies and compile on the game server.
* You can run custom commands to set up your bot by including an `install.sh` file alongside `MyBot.{ext}`. This file will be executed and should be a Bash shell script. You must include the shebang line at the top: `#!/bin/bash`.
* For Python, you may install packages using pip, but you may not install to the global package directory. Instead, install packages as follows: `python3.6 -m pip install --system --target . numpy`
