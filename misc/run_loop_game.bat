:loop
halite.exe --replay-directory ../replays/ --width 32 --height 32 --no-timeout "python ../player1/MyBot.py" "python ../player2/MyBot.py"
halite.exe --replay-directory ../replays/ --width 40 --height 40 --no-timeout "python ../player1/MyBot.py" "python ../player2/MyBot.py"
halite.exe --replay-directory ../replays/ --width 48 --height 48 --no-timeout "python ../player1/MyBot.py" "python ../player2/MyBot.py"
halite.exe --replay-directory ../replays/ --width 56 --height 56 --no-timeout "python ../player1/MyBot.py" "python ../player2/MyBot.py"
halite.exe --replay-directory ../replays/ --width 64 --height 64 --no-timeout "python ../player1/MyBot.py" "python ../player2/MyBot.py"
goto loop
