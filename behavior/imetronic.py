from pathlib import Path
from typing import Union
import numpy as np


FAMILIES_NAME = {1: 'LED', 2: 'lever', 3: 'nosepoke', 4: 'distrib', 5: 'lick', 6: 'misc_ca', 7: 'gate', 9: 'zone',
                 10: 'misc', 11: 'stop', 12: 'rfid', 13: 'message', 14: 'rearing', 15: '15: TTL/undocumented'}

EVENT_CODES = {1: {1: 'HLED'}, 2: {}, 3: {}, 4: {}, 5: {}, 
               6: {1: 'INJ1', 2: 'SND', 3: 'WN', 4: 'SHK', 5: 'PUSH', 6: 'TOP', 7: 'INJ2', 8: 'ADC',
                   9: 'SNDpP', 10: 'FL', 11: 'RD',
                   12: 'OD', 13: 'BUL', 14: 'WH', 15: 'DISK', 20: 'tone', 23: 'shock'},  # 20 and 23 are undocumented
               7: {}, 9: {}, 10: {1: 'ON', 5: 'EVT'},
               11: {4: 'stop'}, 15: {1: 'TTL_ON?'}}
# LED codes
for ix in range(2, 9):
    EVENT_CODES[1].update({ix: f'LED{ix-1}'})

# Lever codes
for ix in range(1, 7):
    EVENT_CODES[2].update({ix: f'L{ix}'})

# Nosepokes codes
for ix in range(1, 6):
    EVENT_CODES[3].update({ix: f'NP{ix}'})

# Distributeur codes
for ix in range(1, 13):
    EVENT_CODES[4].update({ix: f'D{ix}'})

# Lick codes
for ix in range(1, 6):
    EVENT_CODES[5].update({ix: f'LK{ix}'})

# Gate codes
for ix in range(1, 13):
    EVENT_CODES[7].update({ix: f'G{ix}'})
    
# Zone codes
for ix in range(1, 14):
    EVENT_CODES[9].update({ix: f'Z{ix}'})
    

class Event:
    def __init__(self, dat_row):
        self._row = dat_row
        self.ts_ms = dat_row[0]
        self.num_id = dat_row[2]
        family_code = dat_row[1]
        self.family = FAMILIES_NAME[family_code]
        self.str_id = EVENT_CODES.get(family_code, {}).get(self.num_id, str(self.num_id))
        self.extra = dat_row[3:]
        self.p, self.v, self.l, self.r, self.t, self.w, self.x, self.y, self.z = self.extra
        self.state = None
        self.rw_count = None
        self.rw_full = None
        self.current_zone = None
        if family_code == 15:
            # TTL ?
            self.state = self.l
        elif family_code == 5:
            # Lick
            self.rw_count = r
            self.rw_full = self.v
        elif family_code == 9:
            self.current_zone = self.v
            self.is_freezing = self.p
        # Often the T value is the total number of X since the beginning of the experiment
        self.total = self.t
        # Often the state (ON, licking, sound on) is stored in the P value
        self.state = self.p

    @property
    def ts_s(self):
        return self.ts_ms / 1000  

    def __str__(self):
        if self.state is not None:
            state = 'ON ' if self.state else 'OFF '
        else:
            state = ''
        s = f'{self.family} - {self.str_id} {state}@ {self.ts_ms}'
        return s

    __repr__ = __str__


class Imetronic:

    def __init__(self, dat_path):
        super().__init__()
        self.events = load_datfile(dat_path)
        ev_by_id = {}
        ts_by_id = {}
        for c_ev in self.events:
            prev_ev = ev_by_id.get(c_ev.str_id, {})
            prev_ev_state = prev_ev.get(c_ev.state, [])
            prev_ev_state.append(c_ev)
            prev_ev[c_ev.state] = prev_ev_state
            ev_by_id[c_ev.str_id] = prev_ev
        self.ts_by_id = {name: {state: [c_ev.ts_ms for c_ev in events] for state, events in all_ev.items() }
                         for name, all_ev in ev_by_id.items()}
        self.ev_by_id = ev_by_id


def load_datfile(dat_path: Union[str, Path]):
    n_header = 0
    with open(dat_path, 'r') as fp:
        for line in fp:
            if line.count('\t') > 3:
                break
            n_header += 1
    raw_events = np.genfromtxt(dat_path, delimiter='\t', skip_header=n_header, dtype=int)
    all_events = [Event(row) for row in raw_events]
    return all_events
    

if __name__ == '__main__':
    ev = Imetronic('/home/remi/Aquineuro/Data/busquets/SPC_neurons/dat/20240206_Julia_SPC_photometry_males_pc1_230048263__01.dat')
