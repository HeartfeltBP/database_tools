import ast
import configparser
import glob
import json
import random
from dataclasses import InitVar, dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from alive_progress import alive_bar

from database_tools.io.data import write_to_json
from database_tools.io.utils import load_valid_records
from database_tools.io.wfdb import get_data_record, get_signal
from database_tools.processing.detect import (detect_flat_lines,
                                              detect_notches, detect_peaks)
from database_tools.processing.metrics import (get_beat_similarity,
                                               get_similarity, get_snr)
from database_tools.processing.modify import align_signals, bandpass
from database_tools.processing.utils import repair_peaks_troughs_idx, window

@dataclass(frozen=True)
class ConfigItem:
    def set_attr(self, config_item):
        for key, value in config_item.items():
            try:
                object.__setattr__(self, key, ast.literal_eval(value))
            except:
                object.__setattr__(self, key, value)


@dataclass(frozen=True)
class ConfigMapper:

    config_path: InitVar[str]
    data = ConfigItem()
    train = ConfigItem()
    model = ConfigItem()
    sweep = ConfigItem()
    deploy = ConfigItem()

    def __post_init__(self, config_path: str):
        config = configparser.ConfigParser()
        config.read(config_path)
        self.data.set_attr(config['DATASET_PIPELINE'])
        self.train.set_attr(config['TRAINING_PIPELINE'])
        self.model.set_attr(config['MODEL'])
        self.sweep.set_attr(config['SWEEP'])
        self.deploy.set_attr(config['DEPLOYMENT'])


@dataclass
class MetricLogger:

    cm: InitVar[ConfigMapper]
    metrics_dict: dict = field(init=False)
    valid_samples: int = 0
    rejected_samples: int = 0
    patient_samples: int = 0
    prev_patient: str = ''

    def __post_init__(self, cm: list):
        self.metrics_dict = {}
        for m in cm.data.metrics:
            self.metrics_dict.update({m: []})
        self.samples_per_file = cm.data.samples_per_file
        self.samples_per_patient = cm.data.samples_per_patient
        self.max_samples = cm.data.max_samples

    def update_metrics(self, metrics: dict) -> None:
        """Update dataset metrics dictionary and return next action.
           0 -> Continue
           1 -> Write data to file and continue.
           2 -> Write remaining data and stop.

        Args:
            metrics (dict): _description_

        Returns:
            _type_: _description_
        """
        patient = metrics['record_path'].split('/')[1]
        if self.prev_patient != patient:
            self.patient_samples = 0
            self.prev_patient = patient
        if metrics['valid']:
            self.valid_samples += 1
            self.patient_samples += 1
        else:
            self.rejected_samples += 1

        for key in self.metrics_dict.keys():
            self.metrics_dict[key].append(metrics[key])

        if (self.valid_samples % self.samples_per_file) == 0:
            if self.valid_samples >= self.max_samples:
                return 2
            else:
                return 1
        else:
            return 0

    def save_stats(self, file_path):
        df = pd.DataFrame(self.metrics_dict)
        df.to_csv(file_path, index=False)
        print(f'Saving dataset stats to {file_path}')


@dataclass
class Dataset:

    data_dir: InitVar[str]
    ppg: np.ndarray = field(init=False)
    vpg: np.ndarray = field(init=False)
    apg: np.ndarray = field(init=False)
    abp: np.ndarray = field(init=False)

    _nframes: int = field(init=False)

    def __post_init__(self, data_dir: str) -> None:
        frames = [pd.read_json(file, lines=True) for file in glob.glob(f'{data_dir}data/lines/*.jsonlines')]
        object.__setattr__(self, '_nframes', len(frames))
        data = pd.concat(frames, ignore_index=True)
        object.__setattr__(self, 'ppg', np.array(data['ppg'].to_list()))
        vpg = np.gradient(self.ppg, axis=1)  # 1st derivative of ppg
        apg = np.gradient(vpg, axis=1) # 2nd derivative of vpg
        object.__setattr__(self, 'vpg', vpg)
        object.__setattr__(self, 'apg', apg)
        object.__setattr__(self, 'abp', np.array(data['abp'].to_list()))
        self.info

    @property
    def info(self):
        print(f'Data was extracted from {self._nframes} JSONLINES files.')
        print(f'The total number of windows is {self.ppg.shape[0]}.')


@dataclass
class Window:

    sig: np.ndarray
    cm: ConfigMapper
    checks: List[str]

    @property
    def _snr_check(self) -> bool:
        self.snr, self.f0 = get_snr(self.sig, low=self.cm.data.freq_band[0], high=self.cm.data.freq_band[1], df=0.2, fs=self.cm.data.fs)
        return self.snr > self.cm.data.snr

    @property
    def _hr_check(self) -> bool:
        return (self.f0 > self.cm.data.hr_freq_band[0]) & (self.f0 < self.cm.data.hr_freq_band[1])

    @property
    def _flat_check(self) -> bool:
        return not detect_flat_lines(self.sig, n=self.cm.data.flat_line_length)

    @property
    def _beat_check(self) -> bool:
        self.beat_sim = get_beat_similarity(
            self.sig,
            troughs=self.troughs,
            fs=self.cm.data.fs,
        )
        return self.beat_sim > self.cm.data.beat_sim

    @property
    def _notch_check(self) -> bool:
        notches = detect_notches(
            self.sig,
            peaks=self.peaks,
            troughs=self.troughs,
        )
        return len(notches) > self.cm.data.min_notches

    @property
    def _bp_check(self) -> bool:
        self.dbp, self.sbp = np.min(self.sig), np.max(self.sig)
        dbp_check = (self.dbp > self.cm.data.dbp_bounds[0]) & (self.dbp < self.cm.data.dbp_bounds[1])
        sbp_check = (self.sbp > self.cm.data.sbp_bounds[0]) & (self.sbp < self.cm.data.sbp_bounds[1])
        return dbp_check & sbp_check

    @property
    def valid(self) -> bool:
        v = [object.__getattribute__(self, '_' + c + '_check') for c in self.checks]
        return np.array(v).all()

    def get_peaks(self, pad_width=40) -> None:
        x_pad = np.pad(self.sig, pad_width=pad_width, constant_values=np.mean(self.sig))
        peaks, troughs = detect_peaks(x_pad).values()
        peaks, troughs = repair_peaks_troughs_idx(peaks, troughs)
        peaks = peaks - pad_width - 1
        self.peaks = [p for p in peaks if p > -1]
        troughs = troughs - pad_width - 1
        self.troughs = [t for t in troughs if t > -1]


def congruency_check(ppg: Window, abp: Window, cm: ConfigMapper) -> Tuple[float, float, bool]:
    """Performs checks between ppg and abp Window objects.

    Args:
        ppg (Window): Object with ppg data.
        abp (Window): Object with abp data.
        cm (ConfigMapper): Config mapping dataclass.

    Returns:
        bool: True if valid, False if not.
    """
    time_sim = get_similarity(ppg.sig, abp.sig)
    ppg_f = np.abs(np.fft.fft(ppg.sig))
    abp_f = np.abs(np.fft.fft(bandpass(abp.sig, low=cm.data.freq_band[0], high=cm.data.freq_band[1], fs=cm.data.fs)))
    spec_sim = get_similarity(ppg_f, abp_f)
    sim_check = (time_sim > cm.data.sim) & (spec_sim > cm.data.sim)
    hr_delta_check = np.abs(ppg.f0 - abp.f0) < cm.data.hr_delta
    congruency_check = sim_check & hr_delta_check
    return {
        'time_sim': time_sim,
        'spec_sim': spec_sim,
        'cong_check': congruency_check
    }


class DatasetFactory():
    def __init__(
        self,
        data_dir,
    ) -> None:
        self._data_dir = data_dir
        self._partner = 'mimic3'
    def run(self):
        cm = ConfigMapper(self._data_dir + '/config.ini')
        metrics_logger = MetricLogger(cm)

        print('Gettings valid segments...')
        valid_records = load_valid_records(self._data_dir + 'valid_records.csv')

        print('Collecting samples...')
        json_data = ''
        for record_path in valid_records:
            data = self._get_data(record_path)
            if data:
                ppg, abp = data[0], data[1]
            else:
                continue

            # Apply bandpass filter to ppg
            ppg = bandpass(ppg, low=cm.data.freq_band[0], high=cm.data.freq_band[1], fs=cm.data.fs)

            overlap = int(cm.data.fs / 2)
            l = cm.data.win_len + overlap
            idx = window(ppg, l, overlap)

            title_len = 50
            with alive_bar(total=len(idx), title_length=title_len, force_tty=True) as bar:
                for i, j in idx:
                    bar.title(f'Rejected: {metrics_logger.rejected_samples} --- Collected: {metrics_logger.valid_samples}'.rjust(title_len))
                    p = ppg[i:j]
                    a = abp[i:j]

                    p, a = align_signals(p, a, win_len=cm.data.win_len, fs=cm.data.fs)

                    # Evaluate ppg
                    p_win = Window(p, cm, cm.data.checks)
                    p_win.get_peaks()
                    p_valid = p_win.valid

                    # Evaluate abp
                    a_win = Window(a, cm, cm.data.checks + ['bp'])
                    a_win.get_peaks()
                    a_valid = a_win.valid

                    # Evaluate ppg vs abp
                    congruence_check = congruency_check(p_win, a_win, cm)
                    valid = p_valid & a_valid & congruence_check['cong_check']

                    metrics = self._compile_window_metrics(record_path, (i, j), valid, p_win, a_win, congruence_check)
                    status = metrics_logger.update_metrics(metrics)

                    if valid:
                        json_data += json.dumps(dict(ppg=p.tolist(), abp=a.tolist())) + '\n'

                        # Write to file when count is reached. 
                        if status in [1, 2]:
                            file_name = str(int(metrics_logger.valid_samples / cm.data.samples_per_file) - 1).zfill(3)
                            file_path = self._data_dir + f'data/lines/{self._partner}_{file_name}.jsonlines'
                            write_to_json(json_data, file_path); json_data = ''
                            if status == 2:
                                metrics_logger.save_stats(self._data_dir + f'{self._partner}_stats.csv')
                                print('Done!')
                                return
                    bar()
                    if metrics_logger.patient_samples == cm.data.samples_per_patient:
                        break
        return

    def _get_valid_segs(self, valid_path):
        valid_df = pd.read_csv(valid_path)

        # Extract data from valid_df and shuffle in the same order
        temp = list(zip(valid_df['id'], valid_df['url']))
        random.shuffle(temp)
        patient_ids, files = [list(t) for t in zip(*temp)]
        patient_ids, files = list(patient_ids), list(files)
        return (patient_ids, files)

    def _get_data(self, path):
        rec = get_data_record(path=path, record_type='waveforms')
        if rec is not None:
            ppg = get_signal(rec, sig='PLETH')
            abp = get_signal(rec, sig='ABP')
            ppg[np.isnan(ppg)] = 0
            abp[np.isnan(abp)] = 0
            return (ppg, abp)
        else:
            return False

    def _compile_window_metrics(self, record_path: str, window_idx: Tuple[int, int], valid, ppg: Window, abp: Window, congruence_check: dict) -> dict:
        metrics = {
            'record_path': record_path,
            'window_idx': window_idx,
            'valid': valid,
            'time_sim': congruence_check['time_sim'],
            'spec_sim': congruence_check['spec_sim'],
            'ppg_snr': ppg.snr,
            'abp_snr': abp.snr,
            'ppg_hr': ppg.f0 * 60,
            'abp_hr': abp.f0 * 60,
            'sbp': abp.sbp,
            'dbp': abp.dbp,
            'ppg_beat_sim': ppg.beat_sim,
            'abp_beat_sim': abp.beat_sim,
            'flat_ppg': ppg._flat_check,
            'flat_abp': abp._flat_check,
            'ppg_notches': ppg._notch_check,
            'abp_notches': abp._notch_check,
        }
        return metrics
