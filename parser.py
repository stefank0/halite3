import json
import os
import zstd

import hlt

ARBITRARY_ID = -1


def parse_replay_file(file):
    with open(file, 'rb') as f:
        return json.loads(zstd.loads(f.read()))


def parse_replay_folder(folder, max_files=None):
    replay_buffer = []
    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".hlt"):
            continue
        elif max_files is not None and len(replay_buffer) >= max_files:
            break
        else:
            replay_buffer.append(parse_replay_file(os.path.join(folder, file_name)))
    return replay_buffer


if __name__ == '__main__':
    parse_replay_file(
        file=r'replays/replay-20181128-225149+0600-1543423752-32-32.hlt'
    )
