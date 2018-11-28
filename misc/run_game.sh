#!/bin/sh

os=${OSTYPE//[0-9.-]*/}

if [ "$os" = "darwin" ]; then
    ./haliteMac --replay-directory ../replays/ -vvv --width 32 --height 32 "python3 ../MyBot.py ..\parameters.yaml" "python3 ../MyBot.py ../parameters.yaml"
else
    ./halite --replay-directory ../replays/ -vvv --width 32 --height 32 "python3 ../MyBot.py ..\parameters.yaml" "python3 ../MyBot.py ../parameters.yaml"
fi

