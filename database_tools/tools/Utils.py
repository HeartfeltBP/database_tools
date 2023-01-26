import os
from datetime import datetime

def get_iso_dt():
    return datetime.today().isoformat().split('T')[0].replace('-', '')

def build_data_directory(repo_dir, partner):
    dt = get_iso_dt()
    data_dir = repo_dir + f'{partner}-data-{dt}/'
    os.chdir(repo_dir)
    os.mkdir(data_dir)
    os.mkdir(data_dir + 'data/')
    os.mkdir(data_dir + 'data/lines')
    os.mkdir(data_dir + 'data/records')
    os.mkdir(data_dir + 'data/records/train')
    os.mkdir(data_dir + 'data/records/val')
    os.mkdir(data_dir + 'data/records/test')
    return data_dir
