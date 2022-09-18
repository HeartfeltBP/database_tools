import os
import pandas as pd
from datetime import date

def build_empty_database(records_path='https://physionet.org/files/mimic3wdb/1.0/RECORDS-adults'):
    path = f'data-{str(date.today())}/'
    os.mkdir(path)
    os.mkdir(path + 'mimic3')
    os.system(f'wget -q -np {records_path} -P {path}')

    used_records_path = path + 'used_records.csv'
    df = pd.DataFrame(columns=['folder'])
    df.loc[0] = ['test']
    df.index.name = 'index'
    df.to_csv(used_records_path)
    return path
