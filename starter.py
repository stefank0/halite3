from subprocess import check_output

halite_exe = r'misc/halite.exe'
dir_replay = r'replays'
mapsize = 32
n_player = 4


def start():
    args = [halite_exe, '-vvv',
            '--replay-directory', dir_replay,
            '--width', str(mapsize),
            '--height', str(mapsize)] + \
           [r'python MyBot.py'] * n_player
    check_output(args).decode("ascii")


if __name__ == '__main__':
        start()
