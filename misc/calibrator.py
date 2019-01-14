import glob
import json
import re, sys
import logging
import click
import yaml
import time
from subprocess import check_output
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('../')
sys.path.append('')

from misc.parse import evaluate_folder
from parameters import CALIBRATION

if not CALIBRATION:
    raise Exception('parameters.CALIBRATION is set to False.')


class Calibrator:
    """Calibrator object to start or load a stopped Calibration of parameterrs for a halite bot.
    params  --> parameters used in calibrate, state of the parameters is not managed in object, but in the yaml files.
    args    --> arguments in halite command
    iter    --> iteration in calibration
    n_xxx   --> number of xxx
    dir     --> directory
    """

    def __init__(self, parameters: list, mapsize=None, n_player=None, n_games=None, n_iter=None, convergence=0.8,
                 pars_reference='parameters.json', dir_replay=r'replays', bot_path=r'MyBot.py', dir_output=None
                 ):
        self.n_iter = n_iter
        self.n_games = n_games
        self.bot_path = bot_path
        self._dir_replay = dir_replay
        self.convergence = convergence
        self._parameters = parameters

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
        self._pars_reference = self.get_parameters_ref()
        self._log = os.path.join(self._dir_output, '.log')
        if self.iter == 0:
            os.mkdir(self._dir_output)
            os.mkdir(self._dir_pars)
            with open(self._log, 'w'):
                pass
            self.set_parameters(self._pars_default_start_file, self._pars_reference)
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
        return 'Calibrator(mapsize={}, n_player={}, n_games={}, n_iter={}, dir_output=r"{}")'.format(self.mapsize, self.n_player, self.n_games, self.n_iter, self._dir_output)

    def set_dir_output(self):
        """Folder where the hlt and errorlogs will be written"""
        return os.path.join(self._dir_replay, 'calibrator_d{}_t{}_p{}_s{}'.format(time.strftime("%Y%m%d"), time.strftime("%H%M%S"), self.n_player, self.mapsize))

    @property
    def args(self):
        """cmd arguments to run a halite game"""
        exe = 'misc/halite' if sys.platform == 'linux' else 'misc\halite.exe'
        args = [exe, '-vvv', '--no-logs', '--no-timeout',
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
    def _pars_default_start_file(self):
        """File with the default parameters, which update each iteration based on the gradients per parameter"""
        return os.path.join(self._dir_pars, 'parameters_default_start.yaml')

    @property
    def _pars_default_file(self):
        """File with the default parameters, which update each iteration based on the gradients per parameter"""
        return os.path.join(self._dir_pars, 'parameters_{:02d}_default.yaml'.format(self.iter))

    @property
    def _pars_high_file(self):
        """File with the parameters and a parameter with one high step"""
        return os.path.join(self._dir_pars, 'parameters_{:02d}_high_{}.yaml'.format(self.iter, self.param))

    @property
    def _pars_low_file(self):
        """File with the parameters and a parameter with one low step"""
        return os.path.join(self._dir_pars, 'parameters_{:02d}_low_{}.yaml'.format(self.iter, self.param))

    @property
    def _dir_iteration(self):
        """Folder for replay and error files for an iteration"""
        return os.path.join(self._dir_output, 'i{:02d}_{}'.format(self.iter, self.param))

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
        if sys.platform == 'linux':
            return 'python3 {} {}'.format(self.bot_path, pars)
        else:
            return 'python {} {}'.format(self.bot_path, pars)

    def start(self):
        """Run game from commandline"""
        while self.iter < self.n_iter:
            logging.info('iter: {:02d}'.format(self.iter))
            self.param_step()
            self.iter += 1
            self.multiplier *= self.convergence
            if self.iter == self.n_iter:
                self.report()
                return 0
            self.set_parameters(self._pars_default_file,
                                self.get_parameters(self._pars_default_file.replace('{:02d}_default'.format(self.iter),
                                                                                    '{:02d}_default'.format(self.iter-1))))

    def param_step(self):
        """Run a single iteration of a list of parameters"""
        for param in self._parameters:
            logging.info('iter: {:02d} param: {}'.format(self.iter, param))
            self.param = param
            if not os.path.exists(self._dir_iteration):
                os.mkdir(self._dir_iteration)
            step_minus = self._pars_default[param] * self.multiplier
            step_plus = self._pars_default[param] * self.multiplier / (1.0 - self.multiplier)
            logging.info('iter: {:02d} param: {} default value: {}'.format(self.iter, param, self._pars_default[self.param]))
            logging.info('iter: {:02d} param: {} stepsize plus: {}'.format(self.iter, param, step_plus))
            logging.info('iter: {:02d} param: {} stepsize minus: {}'.format(self.iter, param, step_minus))
            self.set_parameter(file=self._pars_low_file, step=-step_minus)
            self.set_parameter(file=self._pars_high_file, step=step_plus)
            while len(glob.glob(os.path.join(self._dir_iteration, '*.hlt'))) < self.n_games:
                logging.info('iter: {:02d} param: {} game: {}'.format(self.iter, param, len(os.listdir(self._dir_iteration))+1))
                check_output(self.args).decode("ascii")
            self.set_parameter(file=self._pars_default_file, step=self.evaluate() - self._pars_default[self.param])
        return 0

    def load(self):
        """Load a calibration state"""
        self.param = re.findall('i\d.+', self.latest_iter)[0][3:]
        self.iter = int(re.findall('i\d{2}', self.latest_iter)[0][1:])
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
                'iter: {:02d} param: {} value: {:.2f} result: {:.0f}'.format(self.iter, self.param, self._params[i], result[i]))
        result -= result.max() - 1500
        result[result < 0] = 0
        return float((self.n_player / result.sum() * result * self._params).mean())

    def report(self):
        files = glob.glob(os.path.join(self._dir_pars, '*default.yaml'))
        for param in self._parameters:
            with open(self._pars_default_start_file) as f:
                params = yaml.load(f)
            result = {param: [params[param]], param + '_low': [np.nan], param + '_high': [np.nan]}
            for file in files:
                with open(file.replace('default', 'low_{}'.format(param))) as f:
                    params = yaml.load(f)
                result[param + '_low'].append(params[param])
                with open(file.replace('default', 'high_{}'.format(param))) as f:
                    params = yaml.load(f)
                result[param + '_high'].append(params[param])
                with open(file) as f:
                    params = yaml.load(f)
                result[param].append(params[param])
            df = pd.DataFrame(result)
            fig, ax = plt.subplots()
            df.plot(ax=ax)
            ax.set_xlabel('n_iter')
            ax.set_ylabel(param)
            ax.set_title(param)
            ax.grid()
            fig.savefig(os.path.join(self._dir_pars, 'report_{}'.format(param)))

    @staticmethod
    def set_parameters(file, pars):
        """Set a dict of parameters to be calibrated to file"""
        with open(file, 'w') as f:
            return yaml.dump(pars, f, default_flow_style=False)

    def get_parameters_ref(self):
        """Load mapsize/#players specific parameters from json file."""
        parameters = self.get_parameters_json(self._pars_reference_file)
        return parameters[str(self.n_player)][str(self.mapsize)]

    @staticmethod
    def get_parameters(file):
        """Get a dict of parameters to be calibrated"""
        with open(file) as f:
            return yaml.load(f)

    @staticmethod
    def get_parameters_json(filename):
        """Load content of a json file."""
        with open(filename) as f:
            return json.load(f)


@click.command()
@click.option('--mapsize', default='32', help='Mapsize.', type=click.Choice(['32', '40', '48', '56', '64']))
@click.option('--n_player', default='2', help='Number of players.', type=click.Choice(['2', '4']))
@click.option('--n_games', default='10', help='Number of games in a iteration step.')
@click.option('--n_iter', default='10', help='Number of iterations.')
@click.option('--convergence', default='0.8', help='Convergence rate.')
@click.option('--param', default='', help='Parameter to be trained.')
@click.option('--dir_output', help='Folder of previous calibration in case you want to continue a calibration.')
def main(mapsize, n_player, n_games, n_iter, dir_output, convergence, param):
    parameters = [param] if param else ['return_distance_factor', 'neighbour_profit_factor', 'return_factor', 'expand_edge_cost']
    calibrator = Calibrator(parameters=parameters,
                            mapsize=int(mapsize), n_player=int(n_player), n_games=int(n_games), n_iter=int(n_iter),
                            convergence=float(convergence), dir_output=dir_output)
    calibrator.start()


if __name__ == '__main__':
    main()
