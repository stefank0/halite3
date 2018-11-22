# Schildpad
## TODO
- [ ] TODO - Improve dropoff points (move ship towards best dropoff loc)
- [ ] TODO - Improve decision make ship <Turn 125 every 1000 build ship; estimated return on investment f(n_ships, halite_available, n_players, map)
- [ ] TODO - Optimize vars possibly with own server or ML?
- [ ] TODO - Improve attack and enemy_threat, predict enemy movement (improve on using just mining_probability()). For enemy_threat, only predict movement for nearby enemy ships.
- [ ] TODO - Improve attack. Make sure that we do not follow an enemy ship to their base (the attack should succeed within a couple of turns: ideally our ship is between the enemy ship and their dropoff)
- [ ] TODO - Improve attack. Make sure there is a friendly ship to take the dropped halite from both collided ships.
- [ ] TODO - Improve attack and enemy_threat, keep track of enemy collisions with our own ships and adjust our behavior depending on enemy behavior (tit-for-tat).
- [ ] TODO - Implement loot for 4 players
- [ ] TODO - Aggresiveness higher for 2p when ships and halite > enemy ships and enemy halite

## Bugs
None

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