import os
from database_tools.tools import BuildDatabase

repo_dir = '/media/cam/CAPSTONEDB/database_tools/'
os.chdir(repo_dir)

config = dict(
    low=0.5,
    high=8.0,
    sim1=0.7,
    sim2=0.9,
    snr_t=20,
    hr_diff=1/6,
    f0_low=0.667,
    f0_high=3.0,
)

worker = BuildDatabase(
    output_dir='data-2022-09-23/',
    config=config,
    win_len=1024,
    fs=125,
    samples_per_file=5000,
    max_samples=10000,
    data_dir='physionet.org/files/mimic3wdb/1.0/',
)

worker.run()
