import json
import os

import zstd
import pandas as pd

import hlt

ARBITRARY_ID = -1


def parse_replay_file(file):
    with open(file, 'rb') as f:
        return json.loads(zstd.loads(f.read()))


def parse_final_production(folder):
    """Evaluates the mean relative final_production to the winner of game in a folder, ranging from 0. to 1."""
    result = []
    for file_name in sorted(os.listdir(folder)):
        scores = []
        if not file_name.endswith(".hlt"):
            continue
        else:
            game = parse_replay_file(os.path.join(folder, file_name))
            for player in game['game_statistics']['player_statistics']:
                scores.append(player['final_production'])
        result.append(scores)
    df = pd.DataFrame(result)
    return df.divide(df.max(axis=1), axis=0).mean()


if __name__ == '__main__':
    test = parse_final_production('replays/calibration_20181129/calibrator_p4_s32_i0_endgame')
    print(test)