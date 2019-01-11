CALIBRATION = False

if CALIBRATION:
    import argparse, yaml


    def get_parser():
        """Get ArgumentParser and add argument."""
        parser = argparse.ArgumentParser('Read *pars.yaml and use parameters in MyBot.')
        parser.add_argument('inputfile', type=str, help='YAML input file containing keyword arguments.')
        return parser


    def load_yaml():
        """Load mapsize/#players specific parameters from yaml file."""
        args = get_parser().parse_args()
        with open(args.inputfile) as y:
            parameters = yaml.load(y)
            parameters['inputfile'] = args.inputfile
            return parameters


else:
    import json


    def load_file(filename):
        """Load content of a json file."""
        with open(filename) as f:
            return json.load(f)


    def get_parameters():
        """Load parameters from json file."""
        try:
            return load_file('parameters.json')
        except IOError:
            return load_file('../parameters.json')


    def load_json(game):
        """Load mapsize/#players specific parameters from json file."""
        number_of_players = str(len(game.players))
        map_size = str(game.game_map.height)
        parameters = get_parameters()
        return parameters[number_of_players][map_size]

param = {}


def load_parameters(game):
    """Load mapsize/#players specific parameters from file."""
    if CALIBRATION:
        parameters = load_yaml()
    else:
        parameters = load_json(game)
    param.update(parameters)
