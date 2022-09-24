from database_tools.tools import BuildDatabase

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
    output_dir='',
    config=config,
    win_len=1024,
    fs=125,
    data_dir='physionet.org/files/mimic3wdb/1.0/',
)

worker.run()
