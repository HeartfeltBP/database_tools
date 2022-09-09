import pandas as pd
from csv import writer

def append_patient(pleth_csv,
                   abp_csv,
                   mrn,
                   pleth,
                   abp,
                   n_samples):

    pleth_df = pd.read_csv(pleth_csv, index_col='index')
    abp_df = pd.read_csv(abp_csv, index_col='index')

    if pleth_df['sample_id'].iloc[-1] != abp_df['sample_id'].iloc[-1]:
        raise ValueError('Files for \'pleth\' and \'abp\' end at different samples.')

    start_idx = pleth_df.index.to_list()[-1] + 1
    with open(abp_csv, 'a') as f:
        w = writer(f)
        for i, idx in enumerate(range(start_idx, start_idx + n_samples)):
            sample_id = mrn + '_' + str(i)
            w.writerow([idx, sample_id] + [abp[i, 0], abp[i, 1]])

    with open(pleth_csv, 'a') as f:
        w = writer(f)
        for i, idx in enumerate(range(start_idx, start_idx + n_samples)):
            sample_id = mrn + '_' + str(i)
            w.writerow([idx, sample_id] + list(pleth[i, :]))
    return

def append_sample_count(data_profile_csv, mrn, n_samples):
    with open(data_profile_csv, 'r') as f:
        idx = int(f.readlines()[-1].split(',')[0]) + 1
    with open(data_profile_csv, 'a') as f:
        w = writer(f)
        row = [idx, mrn, n_samples]
        w.writerow(row)    
    return
