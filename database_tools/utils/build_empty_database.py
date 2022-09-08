import pandas as pd
import os

def build_empty_database(repo_dir):
    os.chdir(repo_dir)
    if not os.path.exists('data/'):
        os.mkdir('data/')

    path = 'data/abp_data.csv'
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['sample_id', 'sbp', 'dbp'])
        df.loc[0] = ['test', 0, 0]
        df.index.name = 'index'
        df.to_csv(path)

    path = 'data/pleth_data.csv'
    if not os.path.exists(path):
        columns = ['sample_id'] + [f'sig_{i}' for i in range(625)]
        df = pd.DataFrame(columns=columns)
        df.loc[0] = ['test'] + [0 for i in range(625)]
        df.index.name = 'index'
        df.to_csv(path)

    path = 'data/sample_count_data.csv'
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['mrn', 'n_samples'])
        df.index.name = 'index'
        df.loc[0] = ['test', 0]
        df.to_csv(path)
