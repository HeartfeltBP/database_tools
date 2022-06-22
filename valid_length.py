"""
Author:      Cameron Johnson
Modified:    06/21/2022
Description: Generate csv of all patients ids, segments ids for all
             segments at least 10 min in length (75,000 samples).         
"""
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import runcmd
from wfdb import rdheader

# Only these lines need to be modified.

OUTFILE = 'RECORDS_30_2'
RECORDS = pd.read_csv('MIMIC-III-RECORDS/RECORDS_30_1.csv', names=['ID'])
DIR_NUM = '30/'

valid_patients = []
base_dir = 'physionet.org/files/mimic3wdb/1.0/'
for pid in tqdm(RECORDS['ID']):
    path = base_dir + DIR_NUM + str(pid) + f'/{pid}'

    runcmd('wget -r -np https://' + path + '.hea')
    hea = rdheader(path)
    segments = np.column_stack((np.array(hea.seg_name[1::]), np.array(hea.seg_len[1::]))) # skip layout header
    valid_segments = []
    for sid, length in segments:
        if (int(length) > 75000) & (sid != '~'):
            valid_segments.append(sid)
    if len(valid_segments) != 0:
        valid_segments.insert(0, pid)
        valid_patients.append(valid_segments)
    runcmd('rm ' + path)

with open(f'MIMIC-III-RECORDS/{OUTFILE}.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(valid_patients)
f.close()
