import time
from subprocess import check_output
import os

from tqdm import tqdm


class Calibrator:
    def __init__(self, mapsize, n_player, n_games, n_iter, dir_replay):
        self.step = 0
        self.bot = r'python ..\MyBot.py {}'
        self.f_pars = '..\parameters.yaml'

        self.mapsize = mapsize
        self.n_player = n_player
        self.n_games = n_games
        self.n_iter = n_iter
        self._dir_replay = dir_replay
        self._dir_output = self.set_dir_output()

    def set_dir_output(self):
        """Folder where the hlt and errorlogs will be written"""
        return os.path.join(self._dir_replay,
                            f'{time.strftime("%Y%m%d_%H%M%S")}_'
                            f'P{self.n_player}_'
                            f'S{self.mapsize}')

    @property
    def args(self):
        return ['halite.exe', '-vvv',
                '--replay-directory', self._dir_output,
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
        os.mkdir(self._dir_output)
        os.mkdir(os.path.join(self._dir_output, 'pars'))

        # read reference parameter yaml
        iteration = 0
        while iteration < self.n_iter:
            # initialize parameters for each step
            # run n games
            for _ in tqdm(range(self.n_games), total=self.n_games):
                check_output(self.args).decode("ascii")
            # evaluate game results
            # determine new stepsize
            iteration += 1
        return

    def set_parameters(self):
        pass


if __name__ == '__main__':
    calibrator = Calibrator(mapsize=32, n_player=4, n_games=100, n_iter=5, dir_replay=r'..\replays')
    calibrator.start()
