import os
from database_tools.tools import BuildDatabase

repo_dir = '/path/to/repository/'
os.chdir(repo_dir)

config = dict(
    low=0.5,
    high=8.0,
    sim1=0.6,
    df=0.2,
    snr_t=2.0,
    hr_diff=1/6,
    f0_low=0.667,
    f0_high=3.0,
    abp_min_bounds=[40, 100],
    abp_max_bounds=[70, 190],
)

worker = BuildDatabase(
    output_dir='/path/to/data/dir/',
    config=config,
    win_len=256,
    fs=125,
    samples_per_file=2500,
    samples_per_patient=500,
    max_samples=300000,
    data_dir='physionet.org/files/mimic3wdb/1.0/',
)

worker.run()
