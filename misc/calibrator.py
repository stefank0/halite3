import time
from subprocess import check_output
import os


class Calibrator:
    def __init__(self, mapsize, n_player, dir_replay):
        self.mapsize = mapsize
        self.n_player = n_player
        self.dir_replay = dir_replay
        self.step = 0
        self.bot = r'python ..\MyBot.py {}'
        self.f_pars = '..\parameters.yaml'

    def update_parameter(self):
        pass

    @property
    def output_path(self):
        return os.path.join(self.dir_replay,
                            f'{time.strftime("%Y%m%d_%H%M%S")}_'
                            f'P{self.n_player}_'
                            f'S{self.mapsize}')

    @property
    def args(self):
        return ['halite.exe', '-vvv',
                '--replay-directory', self.output_path,
                '--width', str(self.mapsize),
                '--height', str(self.mapsize),
                self.set_bot(self.f_pars),
                self.set_bot(self.f_pars),
                self.set_bot(self.f_pars),
                self.set_bot(self.f_pars),
                ]

    def set_bot(self, pars):
        return self.bot.format(pars)

    def start(self):
        """Run game from commandline"""
        os.mkdir(self.output_path)
        self.set_parameters()
        check_output(self.args).decode("ascii")
        
        return

    def set_parameters(self):
        pass


if __name__ == '__main__':
    calibrator = Calibrator(mapsize=32, n_player=4, dir_replay=r'..\replays')
    calibrator.start()
