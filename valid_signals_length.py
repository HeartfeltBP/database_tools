"""
Author:      Cameron Johnson
Modified:    06/21/2022
Description: Generate csv of all patients ids, segments ids for all
             segments at least 10 min in length (75,000 samples) that
             have PLETH & ABP.         
"""
import csv
from tqdm import tqdm
from utils import runcmd
from wfdb import rdheader

with open('MIMIC-III-RECORDS/RECORDS_30_2.csv', 'r') as f:
    x = f.read().splitlines()
f.close()

OUTFILE = 'RECORDS_30_3'
RECORDS = [i.split(',') for i in x]
DIR_NUM = '30/'

valid_patients = []
base_dir = 'physionet.org/files/mimic3wdb/1.0/'
for rec in RECORDS:
    pid = rec[0]
    valid_segments = []
    for sid in rec[1::]:
        path = base_dir + DIR_NUM + str(pid) + f'/{sid}'

        runcmd('wget -r -np https://' + path + '.hea')
        hea = rdheader(path)
        if ('PLETH' in hea.sig_name) & ('ABP' in hea.sig_name):
            valid_segments.append(sid)
        runcmd('rm ' + path)
    if len(valid_segments) != 0:
        valid_segments.insert(0, pid)
        valid_patients.append(valid_segments)

with open(f'MIMIC-III-RECORDS/{OUTFILE}.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(valid_patients)
f.close()
