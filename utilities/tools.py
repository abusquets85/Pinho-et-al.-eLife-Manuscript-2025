from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import median_absolute_deviation as mad
except ImportError:
    from scipy.stats import median_abs_deviation as mad 
    
def full_clean_ax(ax):
    # Hide the spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticklabels([])


def clean_ax(ax):
    """
    to improve the display of the axes in our figures"""
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def zscore_snippets(time, snippets):
    if snippets.ndim < 2:
        return np.array([])
    is_bsl = time < 0
    m = np.median(snippets[:, is_bsl], axis=1)
    s = mad(snippets[:, is_bsl], axis=1)
    snippets = ((snippets.T - m) / s).T
    return snippets


def zscore(signal):
    m = np.median(signal)
    sd = mad(signal)
    z_signal = (signal - m ) / sd
    return z_signal


def prime_factors(n: int):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def num_subplots(n: int):
    if n == 0:
        return [0, 0]
    if n == 1:
        return [1, 1]
    f = prime_factors(n)
    if len(f) > 1:
        while len(f) > 2:
            if len(f) >= 4:
                f[0] = f[0] * f[-2]
                f[1] = f[1] * f[-1]
                f = f[:-2]
            else:
                f[0] = f[0] * f[1]
                f = [f[0], f[-1]]
        f.sort()
    else:
        f = [1, f[0]]
    if f[1] / f[0] > 2.5 and n > 4:
        return num_subplots(n+1)
    return f


def load_tags(data_path, tag_rel_path):
    data_path = Path(data_path)
    tags = {}
    with open(tag_rel_path, 'r') as tf:
        for line in tf:
            content = line.strip().split(',')
            abs_path = data_path / content[0]
            tags[str(abs_path)] = content[1:]
    return tags


def parse_path_photometry(fluorescence_path):
    fluorescence_path = Path(fluorescence_path)
    # date, time = fluorescence_path.parent.name.split('-')
    name_session = fluorescence_path.parent.parent.name
    name_animal = fluorescence_path.parent.parent.parent.name
    info = {}
    try:
        info = dict([item.split('-') for item in name_session.split('_')])
        info.update(dict([item.split('-') for item in name_animal.split('_')]))
    except ValueError:
        return info
    # info['date'] = date
    # info['time'] = time

    return info


def split_stimuli(recording):
    new_events = {}
    for ev_name, ev in recording['events'].items():
        for state, ts in ev.items():
            for ix, c_ts in enumerate(ts):
                pres_name = f'{ev_name}_{ix}'
                prev_ev = new_events.get(pres_name, {})
                prev_ev_state = prev_ev.get(state, [])
                prev_ev_state.append(c_ts)
                prev_ev[state] = prev_ev_state
                new_events[pres_name] = prev_ev
    new_events = {name: {state: np.array(t) for state, t in ev.items()}
                  for name, ev in new_events.items()}
    recording['events'] = new_events

    return recording


def merge_infos(rec, field_sep='_', value_sep='-'):
    str_infos = field_sep.join([f'{k}{value_sep}{v}' for k, v in rec['infos'].items()])
    return str_infos


def save_excel_sheet(dataframe, sheet_name, excel_save_path):
    if excel_save_path is not None:
        excel_save_path = Path(excel_save_path)
        if excel_save_path.exists():
            with pd.ExcelWriter(excel_save_path, mode='a', if_sheet_exists='replace') as writer:
                dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
        else:
            dataframe.to_excel(excel_save_path, index=False, sheet_name=sheet_name)
