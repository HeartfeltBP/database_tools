from database_tools.tools import BuildDatabase
from database_tools.preprocessing.utils import ConfigMapper

data_dir = '/home/cam/Documents/database_tools/data/mimic3-data-20230220/'

config = dict(
    checks=['snr', 'hr', 'beat', 'amp', 'flat'],
    fs=125,                                 # sampling frequency
    win_len=256,                            # window length
    freq_band=[0.5, 8.0],                   # bandpass frequencies
    sim=0.6,                                # similarity threshold
    snr=2.0,                                # SNR threshold
    hr_freq_band=[0.667, 3.0],              # valid heartrate frequency band in Hz
    hr_delta=1/6,                           # maximum heart rate difference between ppg, abp
    dbp_bounds=[20, 130],                   # upper and lower threshold for DBP
    sbp_bounds=[50, 225],                   # upper and lower threshold for SBP
    flat_line_length=10,                    # max length of flat lines
    ppg_amp=0.7,                            # minimum amplitude of ppg wave
    windowsize=1,                           # windowsize for rolling mean
    ma_perc=20,                             # multiplier for peak detection
    beat_sim=0.2,                           # lower threshold for beat similarity
)

cm = ConfigMapper(config=config)

bd = BuildDatabase(
    data_dir=data_dir,
    samples_per_file=2500,
    samples_per_patient=500,
    max_samples=200000,
)

bd.run(cm)
