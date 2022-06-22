"""
Author:      Cameron Johnson
Modified:    06/21/2022
Description: Generate csv of all patients in target directory with 
             both PLETH and ABP waveforms in their layout.hea file.
"""
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import runcmd
from wfdb import rdheader

# Only these lines need to be modified.

OUTFILE = 'RECORDS_30_2'
RECORDS = np.array([int(i.strip('/')) for i in pd.read_csv('MIMIC-III-RECORDS/RECORDS_30', names=['ID'])['ID']])
DIR_NUM = '30/'

valid_patients = []
base_dir = 'physionet.org/files/mimic3wdb/1.0/'
for pid in tqdm(RECORDS):
    path = base_dir + DIR_NUM + str(pid) + f'/{pid}_layout'
    runcmd('wget -r -np https://' + path + '.hea')
    hea = rdheader(path)
    if ('PLETH' in hea.sig_name) & ('ABP' in hea.sig_name):
        valid_patients.append(pid)
    runcmd('rm ' + path)

valid_patients = [[i] for i in valid_patients]
with open(f'MIMIC-III-RECORDS/{OUTFILE}.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(valid_patients)
f.close()
