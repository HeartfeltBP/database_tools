import random

def load_valid_records(file_path: str, shuffle: bool = True) -> list:
    with open(file_path, 'r') as f:
        vr = [line.strip('\n') for line in f]
    if shuffle:
        random.shuffle(vr)
    return vr
