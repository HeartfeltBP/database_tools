import pandas as pd
from csv import writer

def compile_data(sample_count_data_path,
                 pleth_data_path,
                 abp_data_path,
                 mrn,
                 pleth,
                 abp,
                 n_samples):

    with open(pleth_data_path, 'r') as f:
        last_pleth_idx = int(f.readlines()[-1].split(',')[0])
    
    with open(abp_data_path, 'r') as f:
        last_abp_idx = int(f.readlines()[-1].split(',')[0])

    if last_pleth_idx == last_abp_idx:
        start_idx = last_pleth_idx + 1
    else:
        raise ValueError('pleth and abp data end at different indices')

    with open(sample_count_data_path, 'r') as f:
        last_patient_idx = int(f.readlines()[-1].split(',')[0])

    with open(sample_count_data_path, 'a') as f:
        w = writer(f)
        w.writerow([last_patient_idx + 1, mrn, n_samples])

    if n_samples > 0:
        with open(abp_data_path, 'a') as f:
            w = writer(f)
            for i, idx in enumerate(range(start_idx, start_idx + n_samples)):
                sample_id = mrn + str(n_samples)
                w.writerow([idx, sample_id, abp[i, 0], abp[i, 1]])

        with open(pleth_data_path, 'a') as f:
            w = writer(f)
            for i, idx in enumerate(range(start_idx, start_idx + n_samples)):
                sample_id = mrn + str(n_samples)
                w.writerow([idx, sample_id] + pleth[i, :])

def compile_patient(sample_count_data_path, mrn):
    with open(sample_count_data_path, 'r') as f:
        last_patient_idx = int(f.readlines()[-1].split(',')[0])

    with open(sample_count_data_path, 'a') as f:
        w = writer(f)
        w.writerow([last_patient_idx, mrn, 0])
