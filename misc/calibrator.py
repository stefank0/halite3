from datetime import datetime as dt
from subprocess import check_output
import os


class Starter:
    def __init__(self, mapsize, n_player):
        self.mapsize = mapsize
        self.n_player = n_player
        self.dir_replay = r'..\replays'
        # self.parameters =

    def update_parameter(self):
        pass

    def start(self):
        """Run game from commandline"""
        dir_output = os.path.join(self.dir_replay, f'{dt.now}_{self.n_player}_{self.mapsize}')
        print(dir_output)
        os.mkdir(dir_output)
        args = ['halite.exe', '-vvv',
                '--replay-directory', dir_output,
                '--width', str(self.mapsize),
                '--height', str(self.mapsize)] + \
               [r'python ..\MyBot.py ..\parameters.yaml'] * self.n_player
        check_output(args).decode("ascii")


if __name__ == '__main__':
    starter = Starter(mapsize=32, n_player=4)
    starter.start()
