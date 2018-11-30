import yaml
import time
from subprocess import check_output
import os

from tqdm import tqdm


class Calibrator:
    """Calibrator object to start or load a stopped Calibration of parameterrs for a halite bot.
    pars --> parameters
    args --> arguments
    iter --> iteration
    n    --> number
    dir  --> directory
    State of the parameters not managed in object, but in the yaml files.
    """

    def __init__(self, mapsize, n_player, n_games, n_iter,
                 pars_reference='..\parameters.yaml', dir_replay=r'..\replays'):
        self.param = None
        self.iter = 0
        self.bot = r'python ..\MyBot.py {}'

        self.mapsize = mapsize
        self.n_player = n_player
        self.n_games = n_games
        self.n_iter = n_iter
        self._dir_replay = os.path.join(dir_replay, f'calibration_{time.strftime("%Y%m%d")}')
        self._pars_reference_file = pars_reference
        self._pars_reference = self.get_parameters(self._pars_reference_file)
        self._pars_default = self.get_parameters(self._pars_reference_file)

        if not os.path.exists(self._dir_replay):
            os.mkdir(self._dir_replay)

    @property
    def _dir_output(self):
        """Folder where the hlt and errorlogs will be written"""
        return os.path.join(self._dir_replay,
                            f'calibrator_'
                            f'p{self.n_player}_'
                            f's{self.mapsize}_'
                            f'i{self.iter}_'
                            f'{self.param}')

    @property
    def args(self):
        """cmd arguments to run a halite game"""
        args = ['halite.exe', '-vvv',
                '--replay-directory', self._dir_output,
                '--width', str(self.mapsize),
                '--height', str(self.mapsize)]
        if self.n_player == 2:
            return args + [self.get_bot(self._pars_high_file), self.get_bot(self._pars_high_file)]
        elif self.n_player == 4:
            return args + [self.get_bot(self._pars_reference_file),
                           self.get_bot(self._pars_default_file),
                           self.get_bot(self._pars_low_file),
                           self.get_bot(self._pars_high_file)]
        raise NotImplementedError

    @property
    def _pars_default_file(self):
        """File with the default parameters, which update each iteration based on the gradients per parameter"""
        return os.path.join(self._dir_output, f'parameters_{self.iter}_default.yaml')

    @property
    def _pars_high_file(self):
        """File with the parameters and a parameter with one high step"""
        return os.path.join(self._dir_output, f'parameters_{self.iter}_high_{self.param}.yaml')

    @property
    def _pars_low_file(self):
        """File with the parameters and a parameter with one low step"""
        return os.path.join(self._dir_output, f'parameters_{self.iter}_low_{self.param}.yaml')

    def get_bot(self, pars):
        """cmd argument to the a single bot with certain parameters in a yaml file"""
        return self.bot.format(pars)

    def start(self):
        """Run game from commandline"""
        while self.iter < self.n_iter:
            for param in self._pars_reference.keys():
                self.param = param
                os.mkdir(self._dir_output)
                self.set_parameters(self._pars_default_file, self._pars_default)

                self.set_parameter(file=self._pars_low_file, step=-self.step)
                self.set_parameter(file=self._pars_high_file, step=self.step)
                for _ in tqdm(range(self.n_games), total=self.n_games):
                    check_output(self.args).decode("ascii")
            # evaluate game results: gradient=
            # determine gradient and step of pamam: step = +/-gradient * reference_params[param]
            # update default parameters
            self.iter += 1
        return 0

    @property
    def step(self):
        if self.iter == 0:
            return self._pars_default[self.param] * 0.5
        else:
            raise NotImplementedError

    def load(self, folder):
        """Load a calibration state"""
        raise NotImplementedError

    @staticmethod
    def get_parameters(file):
        """Get a dict of parameters to be calibrated"""
        with open(file) as f:
            return yaml.load(f)

    @staticmethod
    def set_parameters(file, pars):
        """Set a dict of parameters to be calibrated to file"""
        with open(file, 'w') as f:
            return yaml.dump(pars, f, default_flow_style=False)

    def set_parameter(self, file, step=0.):
        """Set the parameters to be used in a bot"""
        pars = self._pars_default.copy()
        pars[self.param] = pars[self.param] + step
        return self.set_parameters(file, pars)

    def reset_parameters(self, file):
        """Reset the parameters back to the default parameters"""
        with open(file, 'w') as f:
            return yaml.dump(self._pars_default, f, default_flow_style=False)


if __name__ == '__main__':
    # Run the calibration
    calibrator = Calibrator(mapsize=32, n_player=4, n_games=100, n_iter=1)
    calibrator.start()
