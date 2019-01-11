import re
import logging
import click
import yaml
import time
from subprocess import check_output
import os

import numpy as np

from misc.parse import evaluate_folder


class Calibrator:
    """Calibrator object to start or load a stopped Calibration of parameterrs for a halite bot.
    params  --> parameters used in calibrate, state of the parameters is not managed in object, but in the yaml files.
    args    --> arguments in halite command
    iter    --> iteration in calibration
    n_xxx   --> number of xxx
    dir     --> directory
    """

    def __init__(self, mapsize=None, n_player=None, n_games=None, n_iter=None, convergence=0.8,
                 pars_reference='parameters.yaml', dir_replay=r'replays', bot_path=r'MyBot.py', dir_output=None):
        self.n_iter = n_iter
        self.n_games = n_games
        self.bot_path = bot_path
        self._dir_replay = dir_replay
        self.convergence = convergence

        if dir_output:
            self._dir_output = dir_output
            self.load()
            self.isload = True
        else:
            self.param = None
            self.iter = 0
            self.mapsize = mapsize
            self.n_player = n_player
            self.multiplier = 0.75 if self.n_player == 4 else 0.33
            self._dir_output = self.set_dir_output()
            self.isload = False
        self._dir_pars = os.path.join(self._dir_output, 'pars')
        self._pars_reference_file = pars_reference
        self._pars_reference = self.get_parameters(self._pars_reference_file)
        self._log = os.path.join(self._dir_output, '.log')
        if self.iter == 0:
            os.mkdir(self._dir_output)
            os.mkdir(self._dir_pars)
            with open(self._log, 'w'):
                pass
            self.set_parameters(self._pars_default_file, self._pars_reference)
        logging.basicConfig(filename=os.path.join(self._dir_output, '.log'),
                            level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filemode='a+')
        if self.iter > 0:
            logging.info('-----load and continue-----')

    def __repr__(self):
        """Representation of Calibrator object"""
        return f'Calibrator(mapsize={self.mapsize}, n_player={self.n_player}, n_games={self.n_games}, ' \
               f'n_iter={self.n_iter}, dir_output=r"{self._dir_output}")'

    def set_dir_output(self):
        """Folder where the hlt and errorlogs will be written"""
        return os.path.join(self._dir_replay,
                            f'calibrator_'
                            f'd{time.strftime("%Y%m%d")}_'
                            f't{time.strftime("%H%M%S")}_'
                            f'p{self.n_player}_'
                            f's{self.mapsize}')

    @property
    def args(self):
        """cmd arguments to run a halite game"""
        args = ['misc\halite.exe', '-vvv', '--no-logs', '--no-timeout',
                '--replay-directory', self._dir_iteration,
                '--width', str(self.mapsize),
                '--height', str(self.mapsize)]
        if self.n_player == 2:
            return args + [self.get_bot(self._pars_low_file), self.get_bot(self._pars_high_file)]
        elif self.n_player == 4:
            return args + [self.get_bot(self._pars_default_file),
                           self.get_bot(self._pars_low_file),
                           self.get_bot(self._pars_high_file),
                           self.get_bot(self._pars_default_file)]

    @property
    def _pars_default_file(self):
        """File with the default parameters, which update each iteration based on the gradients per parameter"""
        return os.path.join(self._dir_pars, f'parameters_{self.iter}_default.yaml')

    @property
    def _pars_high_file(self):
        """File with the parameters and a parameter with one high step"""
        return os.path.join(self._dir_pars, f'parameters_{self.iter}_high_{self.param}.yaml')

    @property
    def _pars_low_file(self):
        """File with the parameters and a parameter with one low step"""
        return os.path.join(self._dir_pars, f'parameters_{self.iter}_low_{self.param}.yaml')

    @property
    def _dir_iteration(self):
        """Folder for replay and error files for an iteration"""
        return os.path.join(self._dir_output, f'i{self.iter}_{self.param}')

    @property
    def _pars_default(self):
        """Parameters of the high iteration"""
        return self.get_parameters(self._pars_default_file)

    @property
    def _pars_low(self):
        """Parameters of the high iteration"""
        return self.get_parameters(self._pars_low_file)

    @property
    def _pars_high(self):
        """Parameters of the high iteration"""
        return self.get_parameters(self._pars_high_file)

    @property
    def _params(self):
        """Get list of the parameters used in the iteration"""
        params = []
        if self.n_player == 2:
            params.append(self._pars_low[self.param])
            params.append(self._pars_high[self.param])
        elif self.n_player == 4:
            params.append(self._pars_default[self.param])
            params.append(self._pars_low[self.param])
            params.append(self._pars_high[self.param])
            params.append(self._pars_default[self.param])
        return np.array(params)

    def get_bot(self, pars):
        """cmd argument to the a single bot with certain parameters in a yaml file"""
        return f'python {self.bot_path} {pars}'

    def start(self):
        """Run game from commandline"""
        while self.iter < self.n_iter:
            logging.info(f'iter: {self.iter:02d}')
            if self.isload:
                params = list(self._pars_reference.keys())
                params = params[[i for i, x in enumerate(params) if x == self.param][0]:]
                self.param_step(params)
                self.isload = False
            else:
                self.param_step(self._pars_reference.keys())
            self.iter += 1
            self.multiplier *= self.convergence
            self.set_parameters(self._pars_default_file,
                                self.get_parameters(self._pars_default_file.replace(f'{self.iter}_default',
                                                                                    f'{self.iter-1}_default')))
        return 0

    def param_step(self, params):
        """Run a single iteration of a list of parameters"""
        for param in params:
            logging.info(f'iter: {self.iter:02d} param: {param}')
            if 'mean_halite' not in param:
                continue
            self.param = param
            if not os.path.exists(self._dir_iteration):
                os.mkdir(self._dir_iteration)
            step_minus = self._pars_default[param] * self.multiplier
            step_plus = self._pars_default[param] * self.multiplier / (1.0 - self.multiplier)
            logging.info(f'iter: {self.iter:02d} param: {param} default value: {self._pars_default[self.param]}')
            logging.info(f'iter: {self.iter:02d} param: {param} stepsize plus: {step_plus}')
            logging.info(f'iter: {self.iter:02d} param: {param} stepsize minus: {step_minus}')
            self.set_parameter(file=self._pars_low_file, step=-step_minus)
            self.set_parameter(file=self._pars_high_file, step=step_plus)
            while len(os.listdir(self._dir_iteration)) < self.n_games:
                logging.info(f'iter: {self.iter:02d} param: {param} game: {len(os.listdir(self._dir_iteration))+1}')
                check_output(self.args).decode("ascii")
            self.set_parameter(file=self._pars_default_file, step=self.evaluate() - self._pars_default[self.param])

    def load(self):
        """Load a calibration state"""
        self.param = re.findall('i\d.+', self.latest_iter)[0][3:]
        self.iter = int(re.findall('i\d', self.latest_iter)[0][1:])
        self.mapsize = int(re.findall('s\d{2}', self._dir_output)[0][1:])
        self.n_player = int(re.findall('p\d', self._dir_output)[0][1:])
        self.multiplier = 0.75 if self.n_player == 4 else 0.33
        for i in range(self.iter):
            self.multiplier *= self.convergence

    @property
    def latest_iter(self):
        """Load latest iteration folder to continue later on"""
        files = os.listdir(self._dir_output)
        paths = [os.path.join(self._dir_output, basename) for basename in files]
        return max(paths, key=os.path.getctime)

    def set_parameter(self, file, step):
        """Set the parameters to be used in a bot"""
        pars = self._pars_default.copy()
        pars[self.param] = pars[self.param] + step
        return self.set_parameters(file, pars)

    def reset_parameters(self, file):
        """Reset the parameters back to the default parameters"""
        with open(file, 'w') as f:
            return yaml.dump(self._pars_default, f, default_flow_style=False)

    def result(self):
        return evaluate_folder(self._dir_iteration)

    def evaluate(self):
        """Evaluates the result of the iteration step in the calibration"""
        result = self.result()
        for i in range(len(result)):
            logging.info(
                f'iter: {self.iter:02d} param: {self.param} value: {self._params[i]:.2f} result: {result[i]:.0f}')
        result[result < result.max()] = 0
        return float((self.n_player / result.sum() * result * self._params).mean())

    @staticmethod
    def set_parameters(file, pars):
        """Set a dict of parameters to be calibrated to file"""
        with open(file, 'w') as f:
            return yaml.dump(pars, f, default_flow_style=False)

    @staticmethod
    def get_parameters(file):
        """Get a dict of parameters to be calibrated"""
        with open(file) as f:
            return yaml.load(f)


@click.command()
@click.option('--mapsize', default='32', help='Mapsize.', type=click.Choice(['32', '40', '48', '56', '64']))
@click.option('--n_player', default='2', help='Number of players.', type=click.Choice(['2', '4']))
@click.option('--n_games', default='10', help='Number of games in a iteration step.')
@click.option('--n_iter', default='10', help='Number of iterations.')
@click.option('--convergence', default='0.8', help='Convergance rate.')
@click.option('--dir_output', help='Folder of previous calibration in case you want to continue a calibration.')
def main(mapsize, n_player, n_games, n_iter, dir_output, convergence):
    calibrator = Calibrator(mapsize=int(mapsize), n_player=int(n_player), n_games=int(n_games), n_iter=int(n_iter),
                            convergence=float(convergence), dir_output=dir_output)
    calibrator.start()


if __name__ == '__main__':
    main()
