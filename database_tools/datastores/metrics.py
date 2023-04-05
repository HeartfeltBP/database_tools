import pandas as pd
from database_tools.datastores.signals import Window

class MetricLogger:

    stats: dict = dict(
        mrn=[],
        valid=[],
        time_sim=[],
        spec_sim=[],
        ppg_snr=[],
        abp_snr=[],
        ppg_hr=[],
        abp_hr=[],
        dbp=[],
        sbp=[],
        ppg_beat_sim=[],
        abp_beat_sim=[],
        flat_ppg=[],
        flat_abp=[],
        ppg_notches=[],
        abp_notches=[],
    )
    valid_samples: int = 0
    rejected_samples: int = 0
    patient_samples: int = 0
    prev_mrn: str = ''

    def update_stats(self, mrn: str, is_valid: bool, time_sim: float, spec_sim: float, ppg: Window, abp: Window) -> None:
        if self.prev_mrn != mrn:
            self.patient_samples = 0
            self.prev_mrn = mrn

        if is_valid:
            self.valid_samples += 1
            self.patient_samples += 1
        else:
            self.rejected_samples += 1

        self.stats['mrn'].append(mrn)
        self.stats['valid'].append(is_valid)
        self.stats['time_sim'].append(time_sim)
        self.stats['spec_sim'].append(spec_sim)
        self.stats['ppg_snr'].append(ppg.snr)
        self.stats['abp_snr'].append(abp.snr)
        self.stats['ppg_hr'].append(ppg.f0 * 60)
        self.stats['abp_hr'].append(abp.f0 * 60)
        self.stats['dbp'].append(abp.dbp)
        self.stats['sbp'].append(abp.sbp)
        self.stats['ppg_beat_sim'].append(ppg.beat_sim)
        self.stats['abp_beat_sim'].append(abp.beat_sim)
        self.stats['flat_ppg'].append(ppg._flat_check)
        self.stats['flat_abp'].append(abp._flat_check)
        self.stats['ppg_notches'].append(ppg._notch_check)
        self.stats['abp_notches'].append(abp._notch_check)

    def save_stats(self, path):
        df = pd.DataFrame(self.stats)
        df.to_csv(path, index=False)
        print(f'Saving dataset stats to {path}')
