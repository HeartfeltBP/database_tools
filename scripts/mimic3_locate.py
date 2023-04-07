import pandas as pd
from database_tools.io.wfdb import locate_valid_records
from database_tools.processing.utils import build_data_directory

repo_dir = '/home/cam/Documents/database_tools/'
data_dir = build_data_directory(repo_dir + 'data/', 'mimic3')

valid_records = locate_valid_records(
    signals=['PLETH', 'ABP'],
    min_length=75000,
    n_segments=None,
    shuffle=True,
)
pd.Series(valid_records).to_csv(data_dir + '/valid_records.csv', index=False, header=False)
