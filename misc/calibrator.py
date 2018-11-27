import yaml
import time
from subprocess import check_output
import os

from tqdm import tqdm


class Calibrator:
    """Calibrator object to start or load a stopped Calibration of parameterrs for a halite bot.
    """

    def __init__(self, mapsize, n_player, n_games, n_iter,
                 pars_default='..\parameters.yaml', dir_replay=r'..\replays'):
        self.pars_high = None
        self.pars_low = None
        self.step = 0
        self.bot = r'python ..\MyBot.py {}'

        self.mapsize = mapsize
        self.n_player = n_player
        self.n_games = n_games
        self.n_iter = n_iter
        self._dir_replay = dir_replay
        self.pars_default = pars_default
        self.pars_reference = pars_default
        self._dir_output = self.set_dir_output()
        self._dir_pars = os.path.join(self._dir_output, 'pars')

    def set_dir_output(self):
        """Folder where the hlt and errorlogs will be written"""
        return os.path.join(self._dir_replay,
                            f'{time.strftime("%Y%m%d_%H%M%S")}_'
                            f'P{self.n_player}_'
                            f'S{self.mapsize}')

    @property
    def args(self):
        """cmd arguments to run a halite game"""
        args = ['halite.exe', '-vvv',
                '--replay-directory', self._dir_output,
                '--width', str(self.mapsize),
                '--height', str(self.mapsize)]
        if self.n_player == 2:
            return args + [self.get_bot(self.pars_low), self.get_bot(self.pars_high)]
        elif self.n_player == 4:
            return args + [self.get_bot(self.pars_default),
                           self.get_bot(self.pars_reference),
                           self.get_bot(self.pars_low),
                           self.get_bot(self.pars_high)]
        raise NotImplementedError

    def get_bot(self, pars):
        """cmd argument to the a single bot with certain parameters in a yaml file"""
        return self.bot.format(pars)

    def start(self):
        """Run game from commandline"""
        os.mkdir(self._dir_output)
        os.mkdir(self._dir_pars)
        iteration = 0
        while iteration < self.n_iter:
            for param in self.get_parameters():
                self.set_parameters(param=param, iteration=iteration, step=-20)
                self.set_parameters(param=param, iteration=iteration, step=20)
                for _ in tqdm(range(self.n_games), total=self.n_games):
                    check_output(self.args).decode("ascii")
                # evaluate game results: gradient=
                # determine gradient and step of pamam: step=gradient*
            iteration += 1
        return 0

    def load(self, folder):
        """Load a calibration state"""
        raise NotImplementedError

    def get_parameters(self):
        """Get a list of the parameters to be calibrated"""
        with open(self.pars_reference) as f:
            pars = yaml.load(f)
        return list(pars.keys())

    def set_parameters(self, param, iteration, step):
        """Set the parameters to be used in a bot"""
        with open(self.pars_reference, 'r') as f:
            pars = yaml.load(f)
        pars[param] = pars[param] + step
        if step < 0:
            self.pars_low = os.path.join(self._dir_pars, f'parameters_{param}_{iteration}_low.yaml')
            with open(self.pars_low, 'w') as f_low:
                yaml.dump(pars, f_low, default_flow_style=False)
        elif step > 0:
            self.pars_high = os.path.join(self._dir_pars, f'parameters_{param}_{iteration}_high.yaml')
            with open(self.pars_high, 'w') as f_high:
                yaml.dump(pars, f_high, default_flow_style=False)
        return 0


if __name__ == '__main__':
    calibrator = Calibrator(mapsize=32, n_player=4, n_games=100, n_iter=1)
    calibrator.start()
