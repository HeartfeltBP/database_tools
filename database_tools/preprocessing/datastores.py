import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from database_tools.preprocessing.utils import ConfigMapper
from database_tools.preprocessing.functions import bandpass, get_similarity, get_snr, flat_lines, beat_similarity


@dataclass
class Window:

    sig: np.ndarray
    cm: ConfigMapper
    checks: List[str]

    @property
    def _snr_check(self) -> bool:
        self.snr, self.f0 = get_snr(self.sig, low=self.cm.freq_band[0], high=self.cm.freq_band[1], df=0.2, fs=self.cm.fs)
        return self.snr > self.cm.snr

    @property
    def _hr_check(self) -> bool:
        return (self.f0 > self.cm.hr_freq_band[0]) & (self.f0 < self.cm.hr_freq_band[1])

    @property
    def _flat_check(self) -> bool:
        return not flat_lines(self.sig)

    @property
    def _beat_check(self) -> bool:
        self.beat_sim = beat_similarity(
            self.sig,
            windowsize=self.cm.windowsize,
            ma_perc=self.cm.ma_perc,
            fs=self.cm.fs,
        )
        return self.beat_sim > self.cm.beat_sim

    @property
    def _bp_check(self) -> bool:
        self.dbp, self.sbp = np.min(self.sig), np.max(self.sig)
        dbp_check = (self.dbp > self.cm.dbp_bounds[0]) & (self.dbp < self.cm.dbp_bounds[1])
        sbp_check = (self.sbp > self.cm.sbp_bounds[0]) & (self.sbp < self.cm.sbp_bounds[1])
        return dbp_check & sbp_check

    @property
    def valid(self) -> bool:
        v = [object.__getattribute__(self, '_' + c + '_check') for c in self.checks]
        return np.array(v).all()

def congruency_check(ppg: Window, abp: Window, cm: ConfigMapper) -> Tuple[float, float, bool]:
    """Performs checks between ppg and abp windows.

    Args:
        ppg (Window): Object with ppg data.
        abp (Window): Object with abp data.
        cm (ConfigMapper): Config mapping dataclass.

    Returns:
        bool: True if valid, False if not.
    """
    time_sim = get_similarity(ppg.sig, abp.sig)
    ppg_f = np.abs(np.fft.fft(ppg.sig))
    abp_f = np.abs(np.fft.fft(bandpass(abp.sig, low=cm.freq_band[0], high=cm.freq_band[1], fs=cm.fs)))
    spec_sim = get_similarity(ppg_f, abp_f)
    sim_check = (time_sim > cm.sim) & (spec_sim > cm.sim)
    hr_delta_check = np.abs(ppg.f0 - abp.f0) > cm.hr_delta
    congruency_check = sim_check & hr_delta_check
    return (time_sim, spec_sim, congruency_check)
