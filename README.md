# Schildpad
## TODO
- [ ] TODO - Conquer or reconquer area from enemy (awareness of area)
- [ ] TODO - Dropoff planner
        - don't just look to current positions of ships --> plan ahead  
        - ghost dropoff
- [ ] TODO - Improve decision spawn ship <Turn 200 every 1000 build ship; estimated return on investment f(n_ships, halite_available, n_players, map)
- [ ] TODO - Return edge costs. Use parameters in the calculation: Halite's left on the map, turns left. (Halite / turns left)
- [ ] TODO - Optimise contants: 
    - HLT parser (pip install Zstandard)
    - Use gradient decent algorithm.
    - Define constants per map size and players.
    - One constant/dimension at the time, one step all together.
    - Depending on turn number. 
- [ ] TODO - Improve attack and enemy_threat, predict enemy movement (improve on using just mining_probability()). For enemy_threat, only predict movement for nearby enemy ships. (Machine learning?)
- [ ] TODO - Check qualitative behaviour, check if parameters are broad enough, or that rules needs to reimplemented.
- [ ] TODO - Improve loot:
    - Increase aggressiveness in end game.
    - Multiple strategies
    - Predict success rate of an attack.
    - Improve attack and enemy_threat, keep track of enemy collisions with our own ships and adjust our behavior depending on enemy behavior and (tit-for-tat - only for lower ranked based on mined Halite).
- [ ] TODO - Factor (0. - 1.) number of free turns (endgame)
- [ ] TODO - Improve returning within (dependent on closeness to dropoff in stead of hardcoded numbers

## Bugs
- [ ] Bot is currently not deterministic. 

# Halite III
General information from Halite III adjusted to our project.

## Halite III components
* /docs, contains API of the game of Halite 
* /hlt, contains modifiable helper functions for your bot
* /misc, contains command scripts and executables to run and example game (.bat for windows) (.sh for MacOS, Linux)
* /replays, contains replays and error files
* MyBot.py, schildpad bot
* scheduler.py, module to assign ships to destinations (distance is 0 to mapsize)
* schedule.py, module to make next step schedule (distance is 0 or 1)
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
