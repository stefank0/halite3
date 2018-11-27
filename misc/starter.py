from subprocess import check_output

mapsize = 32
n_player = 4


def start():
    args = ['halite.exe', '-vvv',
            '--replay-directory', r'..\replays',
            '--width', str(mapsize),
            '--height', str(mapsize)] + \
           [r'python ..\MyBot.py ..\parameters.yaml'] * n_player
    check_output(args).decode("ascii")


if __name__ == '__main__':
        start()
