# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:21:36 2022

@author: aconde
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simps
import scipy.interpolate as ip
from tqdm import tqdm

from photometry.plotting import plot_recording
from utilities.colors import wvl_names
from utilities.tools import load_tags, parse_path_photometry, split_stimuli, zscore_snippets, \
    merge_infos


# #########################  Individual file functions #############################
def load_photometry(filepath):
    """
    This function loads one photometry file. Later we will use this 
    function with a for loop to read all the files
    
    Parameters
    ----------
    filepath: Path or str
    
    Returns
    -------
    photo_data: np.ndarray
        one row per timestamp and 5 columns
    ch_num: tuple
        Channel numbers
    ch_wvl: tuple
        All recorded wavelenghts
    """
    with open(filepath, 'r') as fp:
        # no need to specify the line, goes to the first in first execution,
        # next line in the second, etc.
        fp.readline()
        # Gets the header to use later # TimeStamp,Markings,CH1-410,CH1-470,
        header_raw = fp.readline().strip().split(',')
        # This line extracts the header but usually adds an empty column at end,
        # we fix this in next line
        # .strip() eliminates useless white spaces that can be problematic
        # .split(,) extracts the elements separated by ','

    # In this line we stop indenting so header is outside 'with open'
    header = [col_name for col_name in header_raw if
              len(col_name) > 0]  # Only columns with name greater than 0
    n_cols = len(header)  # number of columns in header for later
    n_sig = n_cols - 2  # which columns contains signals (minus time and marker columns)
    # describing the data type of each column,
    # useful when using markers (strings, messy for np.arrays)
    col_dtypes = ['float32', '|S300'] + [
        # 1 float + 1 string + 1 float * number of signal channels
        # (this makes it useful for when we have more signal channels)
        'float32'] * n_sig
    # list of tuples containing column header with corresponding data type (paired with zip)
    custom_dtype = [(name, dtype) for name, dtype in zip(header, col_dtypes)]
    photo_data = np.genfromtxt(filepath, delimiter=',', skip_header=2,
                               dtype=custom_dtype)
    photo_data['TimeStamp'] /= 1000
    # In the variable explorer dtype=custom_dtype returns a single vector of data
    # (timestamps, without header, void2848 type) but this does not matter
    # It will work fine when we reun the other functions

    # Let's guess the number of recording sites and wavelengths
    channels = [h for h in header if h.startswith('CH')]
    ch_split = [ch.split('-') for ch in channels]
    ch_num = tuple(set([int(ch[0][2:]) for ch in ch_split]))
    ch_wvl = tuple(set([int(ch[1]) for ch in ch_split]))
    return photo_data, ch_num, ch_wvl


def get_markers(data, filepath=None):
    """
    Extract markers and their timestamps from raw RWD photometry data

    Parameters
    ----------
    data: np.ndarray
        Raw RWD data as returned by :py:func:`load_rwd`

    filepath: Optional, str or Path
        Photometry file path. Because we could use the Events.csv file in the same folder if it exists

    Returns
    -------
    markers: dict
        Dictionary containing all markers, with all states, and the timestamps associated with each
        state change for each marker. The timestamps are in numpy arrays
        .. code-block::

            In general:
            {Marker: {state: np.ndarray([timestamps]) } }
            Example:
            { 'Input1': { 0: [t1, t2, t3], 1: [t4, t5, t6] }, 'Marker2': {0: [], 1: []} }

    """
    if filepath is not None:
        filepath = Path(filepath)
        ev_path = filepath.parent / 'Events.csv'
        if ev_path.exists():
            return load_event_file(ev_path)
    if 'Markings' in data.dtype.names:
        marker_field = 'Markings'
    else:
        marker_field = 'Events'
    gi = data[marker_field] != b''
    raw_markers = data[marker_field][gi]
    m_time = data['TimeStamp'][gi]

    all_markers = [m.decode().strip(';').split(';') for m in raw_markers]
    marker_ts = {}
    # Une boucle for pour parcourir les frames avec événements avec le timestamp associé
    for t, am in zip(m_time, all_markers):
        # t: timestamp examiné ; am: tous les markers arrivant à ce timestamp
        # Une boucle pour parcourir chaque événement à un timestamp donné
        for m in am:  # m: 'Input1*2*0'
            cm = m.split('*')
            marker_name = cm[0]  # Input1
            marker_state = int(cm[2])  # 0
            prev = marker_ts.get(marker_name, {})  # {0: [], 1: []}
            ts = prev.get(marker_state, [])  # une liste, vide ou non, de timestamps
            ts.append(t)  # [73.999472]
            prev[marker_state] = ts  # Mise à jour du dict du marker actuel
            marker_ts[marker_name] = prev  # Mise à jour du dictionnaire global
    # Double comprehension de dictionnaire pour convertir les listes de timestamps en array
    markers = {marker_name: {state: np.array(ts) for state, ts in states.items()}
               for marker_name, states in marker_ts.items()}

    # calculate marker duration and add to each marker array
    for mark in markers:
        start = markers[mark][0]
        stop = markers[mark][1]
        if len(start) == len(stop) + 1:
            stop = np.append(stop, data['TimeStamp'][-1])
        duration = stop - start
        markers[mark][2] = duration
    return markers


def load_event_file(ev_path):
    """
    Load an Events.csv file created by new version of the RWD software with mor eprecise events timestamps

    Parameters
    ----------
    ev_path: Path
        Events.csv path

    Returns
    -------
    events: dict
        Dictionary containing all markers, with all states, and the timestamps associated with each
        state change for each marker. The timestamps are in numpy arrays
        Same data structure as for the get_markers function (legacy format)

    """
    ev = pd.read_csv(ev_path)
    events = {}
    for ix_row, row in ev.iterrows():
        c_ev_state = events.get(row['Name'], {})
        c_ev_ts = c_ev_state.get(row['State'], [])
        c_ev_ts.append(row['TimeStamp'])
        c_ev_state[row['State']] = c_ev_ts
        events[row['Name']] = c_ev_state
    # Double comprehension de dictionnaire pour convertir les listes de timestamps en array
    events = {ev_name: {state: np.array(ts) / 1000 for state, ts in states.items()}
              for ev_name, states in events.items()}

    return events


def trim_data(data, trim_duration):
    """

     Trim the beginning of the recording to eliminate the first minute
       
     Mask : Array of bool. True codifies the values of 'TimeStamp' we will keep 
     (those over 60 seconds), False the ones we will trimm (those below 60s).
    
     Parameters
     ----------
     data : np.ndarray
     Contains our fluorescence data points.
         trim_duration : Int, optional
     Duration of the period we want to trim in seconds. The default is 60.
    
     Returns
     -------
     trimmed_data : ndarray
         Our data without the first 60 seconds (or whathever we write in 'trim_duration').

    """

    mask = data['TimeStamp'] > trim_duration  # masking the data to ignore the first minute
    trimmed_data = data[:][mask]  # applying mask
    return trimmed_data


def iso_correction(data, chann='CH1', sig_wvl=470, iso_wvl=410):
    """
    least-squares linear fit to the isosb signal to align
    it to the fluo signal before calculating df/f
    source: Lerner et al. 2015, doi.org/10.1016/j.cell.2015.07.014

    Parameters
    ----------
    data : np.ndarray
        We use the isosbestic data['CH1410'] or the fluo signal ['CH1470'] 
    chann: str
        the name of the channel we want to analyze

    Returns
    -------
    iso_scaled: np.ndarray
        Scaled iso

    """
    p = np.polyfit(data[f'{chann}{iso_wvl}'], data[f'{chann}{sig_wvl}'], 1)
    iso_scaled = np.polyval(p, data[f'{chann}{iso_wvl}'])
    # sig = data[f'{chann}470'] - iso_scaled
    return iso_scaled


def box_filter(t, signal, win_size=30):
    """

    Parameters
    ----------
    dt
    signal
    win_size: in seconds

    Returns
    -------

    """

    dt = np.median(np.diff(t))
    n_pts = int(win_size / dt)
    box = np.ones(n_pts) / n_pts
    smoothed_sig = np.convolve(signal, box)
    smoothed_sig = smoothed_sig[:len(signal)]
    return smoothed_sig


def delta_f(data, trim_duration, chann='CH1', wavelength=470, iso_wvl=410):
    """
    Compute the Delta F / F

    Parameters
    ----------
    data
    trim_duration
    chann: str
        Name of the channel to use
    wavelength: int
        Signal wavelength. Either 470 or 560 depending on the fluorophore used
    iso_wvl: int
        Isosbestic wavelength. Either 410 if signal is 470 or 470 if signal is 560.
        If it is set to None, we do not use iso correction, we return raw data instead
        but still trimmed from the first trim_duration seconds

    Returns
    -------
    t_delta: time vector for delta
    delta: our delta f/f data trimmed and smoothed

    """
    # Trim the beginning of recording
    # Fit the isosbestic to the GCaMP
    # Correct the GCaMP using fitted iso
    # Some kind of DF/F computation
    t_data = trim_data(data, trim_duration)  # this is the trimmed data
    if iso_wvl is not None:
        # this is the trimmed fitted isosbestic signal
        iso_scaled = iso_correction(t_data, chann, sig_wvl=wavelength, iso_wvl=iso_wvl)
        sig = t_data[f'{chann}{wavelength}']
        delta = 100 * (sig - iso_scaled) / iso_scaled
    else:
        delta = t_data[f'{chann}{wavelength}']

    return t_data['TimeStamp'], delta


def analyze_mouse(filepath, mouse_name, site_names=None, sensor_names=None, trim_duration=60):
    """
    Load photometry data from the Fluorescence file and organize the signals by recording site
    and sensor and compute all DF/F using the isosbestic signal

    Parameters
    ----------
    filepath : Path 
        The path of the single file to analyse.
    mouse_name : str
        String with the mouse name.(i.e: 'A2')
    site_names: dict, optional
        Dictionary containing channel numbers as keys and  name of the recording sites as values
    sensor_names: dict, Optional
        Describing the sensors used, as a function of their wavelength.
        If let at None will default to
        {410: 'Isosbestic', 470: 'GCaMP', 560: 'RCaMP'}
    trim_duration: float
        Durtion to trim from the beginning of the recording because of bleaching

    Returns
    -------
    result : Dict
        A dictionary containing the information of our animal. If n_chann > 1 it saves
        a dictionary per channel inside the result dict. Take into account when calling 
        data from other functions.

    """
    filepath = Path(filepath)
    if sensor_names is None:
        sensor_names = wvl_names
    data, ch_num, ch_wvl = load_photometry(filepath)
    all_channels = {}
    all_raw = {}
    t_delta = None
    if site_names is None:
        site_names = {ch: ch for ch in ch_num}

    for ch_ix, site in site_names.items():
        if ch_ix not in ch_num:
            print(f'Warning: {ch_ix} is not a valid channel number for file {filepath}')
            continue
        channel_name = f'CH{ch_ix}'
        ch = {}
        raw = {}
        for w in ch_wvl:
            # Here we compute all DF/F by using on physio signal and one iso signal
            raw[w] = data[f'{channel_name}{w}']
            if w == 410:
                # Not processing isosbestic as signal
                continue
            if w == 470 and 410 in ch_wvl:
                # This is a GCaMP recording, compute DF/F
                t_delta, dff = delta_f(data, trim_duration, chann=channel_name, wavelength=w,
                                       iso_wvl=410)
            elif w == 560 and 410 in ch_wvl:
                # Here we record GCaMP + RCaMP, we can not use 470 as the iso signal
                t_delta, dff = delta_f(data, trim_duration, chann=channel_name, wavelength=w,
                                       iso_wvl=410)
            elif w == 560 and 410 not in ch_wvl:
                # We could even check for the existence of the 470 channel
                # Here we record RCaMP only, so we can use 470 as the iso signal
                t_delta, dff = delta_f(data, trim_duration, chann=channel_name, wavelength=w,
                                      iso_wvl=470)
            elif w == 470 and 410 not in ch_wvl:
                # We are recording RCaMP only, so 470 is an isosbestic signal
                # There is no DF/F to be computed using 470
                continue
            else:
                raise ValueError('Something we did not plan for')
            w_name = sensor_names[w]
            ch[f'{w_name}'] = dff

        all_channels[site] = ch
        all_raw[site] = raw

    result = {'mouse': mouse_name, 'channels': all_channels,
              'markers': get_markers(data, filepath), 'time': t_delta, 'raw': all_raw,
              'raw_time': data['TimeStamp'],
              'filepath': filepath}

    return result


def time_interpolation(t_target, target, t_ref):
    # Linear fit to the isosbestic data
    interpolator = ip.interp1d(t_target, target, 'linear',
                               assume_sorted=True, fill_value='extrapolate')
    # Interpolate the ref signal at the signal times
    interp_iso = interpolator(t_ref)
    return interp_iso


def align_fragments(t_frag, all_frag):
    # The inversion of min and max is not a mistake
    # We look for the larger t_start and the smaller t_end, so the interval that all snips cover
    t_start = np.max([t[0] for t in t_frag])
    t_end = np.min([t[-1] for t in t_frag])
    dt = np.min([np.median(np.diff(t)) for t in t_frag])
    t_ref = np.arange(t_start, t_end+dt, dt)
    all_frag_aligned = [time_interpolation(t, snip, t_ref) for t, snip in zip(t_frag, all_frag)]
    all_frag_array = np.array(all_frag_aligned)
    return t_ref, all_frag_array


def get_snippets(t, signal, events, t_left=5., t_right=10.):
    n_events, sel_events, start_ix, stop_ix = get_start_stop_ix_snippets(t, events, t_left, t_right)
    if n_events is None:
        return np.array([]), np.array([]), np.array([])
    all_snips = []
    all_t_snips = []
    for ix in range(n_events):
        c_start = start_ix[ix]
        c_stop = stop_ix[ix] + 1
        sig_slice = signal[c_start:c_stop]
        t_slice = t[c_start:c_stop] - sel_events[ix]
        all_snips.append(sig_slice)
        all_t_snips.append(t_slice)
    t_ref, snippets = align_fragments(all_t_snips, all_snips)

    return t_ref, snippets, sel_events


def get_start_stop_ix_snippets(t, events, t_left, t_right):
    sel_events = events[events > t[0] + t_left]  # [0, 15, 90, 99] -> [15, 90, 99]
    sel_events = sel_events[sel_events < t[-1] - t_right]  # [15, 90]
    n_events = len(sel_events)
    if n_events == 0:
        return None, None, None, None
    starts = sel_events - t_left
    stops = sel_events + t_right
    start_ix = np.searchsorted(t, starts)
    stop_ix = np.searchsorted(t, stops)
    return n_events, sel_events, start_ix, stop_ix


def find_transients(t, signal, k_mad=5, delay_th=1, transient_duration_th=1, height_th=None):
    """

    Parameters
    ----------
    t: np.ndarray
    signal: np.ndarray
    k_mad: float
        Number of MAD above which a peak is considered a transient
        Default to 5.
    delay_th: float
        In seconds, minimum time between two transients
    transient_duration_th: float
        In seconds, minimal transient duration
    height_th: float or None

    Returns
    -------

    """
    dt = np.median(np.diff(t))
    f_signal = signal
    med = np.median(f_signal)
    mad = np.median(np.abs(f_signal - med))
    th = k_mad * mad
    pk, prop = find_peaks(f_signal, prominence=th, distance=int(delay_th / dt),
                          width=int(transient_duration_th / dt), height=height_th)

    return pk, prop


def characterize_recording(t, signal, k_mad=5, height_th: Optional[int] = 1, delay_th=1):
    pk, prop = find_transients(t, signal, k_mad, height_th=height_th, delay_th=delay_th)
    if len(pk) == 0:
        results = {'pk': pk, 'prop': prop, 'frequency': np.nan, 'amplitude': np.nan,
                   'width': np.nan, 'AUC': np.nan, 'transient_ts': pk, 'all_auc': np.array([])}
        return results

    # Changing the way I compute duration to take into account that time might have holes
    # if we only consider  parts of the recording
    dt = np.median(np.diff(t))
    duration = len(t) * dt
    transient_freq = len(pk) / duration
    transient_amp = np.mean(prop['prominences'])
    transient_width = np.mean(prop['widths'])
    left_ix = np.intp(prop['left_ips'])
    right_ix = np.intp(prop['right_ips'])
    transient_auc = [simps(signal[l:r], t[l:r]) for l, r in zip(left_ix, right_ix)]

    results = {'pk': pk, 'prop': prop, 'frequency': transient_freq, 'amplitude': transient_amp,
               'width': transient_width, 'AUC': np.mean(transient_auc), 'transient_ts': t[pk],
               'all_auc': np.array(transient_auc)}
    return results


def auc_snips(t_snips, snips, bsl_win=(-10, 0), resp_win=(0, 10)):
    is_bsl_mask = np.logical_and(t_snips > bsl_win[0], t_snips < bsl_win[1])
    is_resp_mask = np.logical_and(t_snips > resp_win[0], t_snips < resp_win[1])
    bsl = snips[:, is_bsl_mask]
    resp = snips[:, is_resp_mask]
    auc_bsl = simps(bsl, t_snips[is_bsl_mask], axis=1)
    auc_resp = simps(resp, t_snips[is_resp_mask], axis=1)
    return auc_bsl, auc_resp



def load_recording(file_path, rec_sites=None, sensor_names=None, trim_duration_s=30):
    """
    Load a recording, ie all data related to a given trial.

    Parameters
    ----------
    file_path: str or Path
        Path to photometry data
    rec_sites: dict, Optional
        Dictionnary giving names to recording sites
        Example:
        {1: 'dHPC', 2: 'vHPC'}
        If left to None will be {1: 1, 2: 2}
    sensor_names: Optional, dict
        Describing the sensors used, as a function of their wavelength.
        If let at None will default to
        {410: 'Isosbestic', 470: 'GCaMP', 560: 'RCaMP'}
    trim_duration_s: float
        How much time to trim from the beginning of the recording.

    Returns
    -------
    rec: dict
        Contains:
        dff: DeltaF/F
        time: time vector
        events: timestamps of events (from the photometry system)
        infos: dict as returned by parse_path augmented with the "behavior" key that contains
               paths to the DLC and poly files.
    """

    file_path = Path(file_path)
    infos = parse_path_photometry(file_path)
    data = analyze_mouse(file_path, infos.get('mouse', 'unknown'), rec_sites, sensor_names=sensor_names,
                         trim_duration=trim_duration_s)

    rec = {'dff': data['channels'], 'time': data['time'], 'events': data['markers'],
           'infos': infos}
    return rec



def analyse_recording(filepath, rec_sites=None, sensor_names=None,
                      t_left=20, t_right=20, trim_duration_s=60,
                      save_dir=None, close_fig=False, make_fig=False,
                      dff_colors=None, events_colors=None, show_ind=True,
                      tags: Optional[Sequence[str]] = None, split_events=False):
    """
    Entry point to analyse one photometry recording.
    It loads a photometry recording

    Parameters
    ----------
    filepath: str or Path
    rec_sites: dict, Optional
        Dictionnary giving names to recording sites
        Example:
        {1: 'dHPC', 2: 'vHPC'}
        If left to None will be {1: 1, 2: 2}
    sensor_names: Optional, dict
        Describing the sensors used, as a function of their wavelength.
        If let at None will default to
        {410: 'Isosbestic', 470: 'GCaMP', 560: 'RCaMP'}
    t_left: float
        Time in seconds to extract before an event when building a PETH
    t_right: float
        Time in seconds to extract after an event when building a PETH
    trim_duration_s: float
        How much time to trim from the beginning of the recording.
    save_dir: Optional, Path or str
    close_fig: bool
        Shall we close the figure? Useful when plotting lots of data
    make_fig: bool
        Should we produce the figures or just load the data
    dff_colors: dict, optional
        Dictionary specifying the colors of each DF/F per site and sensor
        If left to None (default) colors will be generated automatically
    show_ind: Optional, bool
        Should the individual events be plotted or just the average?
        Defaults to True
    events_colors: dict, optional
        Dictionary specifyinfg the events colors. If left to None, the colors will be picked
        automatically
    tags: Optional, list of str
        List of strings ('good', 'bad') qualifying the signals.
        Bad signals are dropped
        If not provided (is None) then nothing is dropped
        Defaults to None.
    split_events: bool
        Should every presentation of one event be treated as a different event?
        Default to False

    Returns
    -------
    rec:
        As returned from load_recording.
    snippets:
        As returned from extract_all_snippets
    df: pd.DataFrame
        Quantification as returned from pa.characterize_recording but in a dataframe format

    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    rec = load_recording(filepath, rec_sites=rec_sites, sensor_names=sensor_names,
                         trim_duration_s=trim_duration_s)
    # FIXME: TAGS NOT WORKING
    # if tags is not None:
    #     rec['dff'] = {site: data
    #                   for site, data in rec['dff'].items()
    #                   if tags[CH_SITE[site]-1] == 'good'}
    if split_events:
        rec = split_stimuli(rec)

    snippets = extract_all_snippets(rec, t_left, t_right)

    res = {site: {sensor: characterize_recording(rec['time'], sig, height_th=None)
                  for sensor, sig in c_site_sig.items()}
           for site, c_site_sig in rec['dff'].items()}
    if make_fig:
        sig_fig, peth_fig = plot_recording(rec, snippets, dff_colors=dff_colors, show_ind=show_ind,
                                           events_colors=events_colors, save_dir=save_dir)
        if close_fig:
            plt.close(sig_fig)
            plt.close(peth_fig)
    keys = ('frequency', 'amplitude', 'width', 'AUC', 'pk', 'prop', 'transient_ts', 'all_auc')
    all_res = []
    for site, all_data in res.items():
        for sensor, data in all_data.items():
            res_d = {k: data[k] for k in keys}
            res_d['site'] = site
            res_d['sensor'] = sensor
            all_res.append(res_d)

    df = pd.DataFrame(all_res)
    for k, v in rec['infos'].items():
        df[k] = v

    if save_dir is not None:
        filename = merge_infos(rec)
        df.to_csv(save_dir / f"{filename}.csv",
                  index=False)
    return rec, snippets, df


def load_dataset(data_path, rec_sites=None, sensor_names=None,
                 t_left = 2, t_right = 5, trim_duration_s = 30, save_dir = None,
                 tag_filename='tags_rel.csv', show_ind=True,  make_fig = False,
                 drop_bad = True, split_events = False, events_colors=None):
    """
    Analyze all recordings in a folder and its subfolders. It is looking for Fluorescence.csv files

    Parameters
    ----------
    data_path: str or Path
        Path to the directory containing data
    rec_sites: dict, Optional
        Dictionnary giving names to recording sites
        Example:
        {1: 'dHPC', 2: 'vHPC'}
        If left to None will be {1: 1, 2: 2}
    sensor_names: Optional, dict
        Describing the sensors used, as a function of their wavelength.
        If let at None will default to
        {410: 'Isosbestic', 470: 'GCaMP', 560: 'RCaMP'}
    t_left: float
        Time in seconds to extract before an event when building a PETH
    t_right: float
        Time in seconds to extract after an event when building a PETH
    trim_duration_s: float
        How much time to trim from the beginning of the recording.
    save_dir: Optional, Path or str
        Path to save the data
    tag_filename: str, optional
        Name of the file in which tags regarding signal quality are saved
    show_ind: Optional, bool
        Should the individual events be plotted or just the average?
        Defaults to True
    make_fig: bool
        Should we produce the figures or just load the data
    drop_bad: Optional, bool
        Should recordings that are tagged 'bad' be dropped from the dataset?
        Defaults to True
    split_events: bool
        Should every presentation of one event be treated as a different event?
        Default to False

    Returns
    -------
    dataset: List of dict
        Each item is one recording represented as a dictionary with:
        Keys: dff, time, events, infos, snippets and transients
        snippets comes from the get_extract_all_snippets function
        transients from the characterize_recording function
    full_rec_df: pd.DataFrame
        Dataframe containing quantification for the transients of all recordings
    """
    data_path = Path(data_path)
    l_files = list(data_path.rglob('Fluorescence.csv'))
    tag_rel_path = data_path / tag_filename
    if tag_rel_path.exists():
        tags = load_tags(data_path, tag_rel_path)
    else:
        tags = {}
    dataset = []
    l_rec_quantif_df = []
    for c_fp in tqdm(l_files):
        c_tags = tags.get(str(c_fp), None)
        if not drop_bad:
            c_tags = None
        rec, snippets, c_df = analyse_recording(c_fp, rec_sites=rec_sites, sensor_names=sensor_names,
                                                t_left=t_left, t_right=t_right, trim_duration_s=trim_duration_s,
                                                save_dir = save_dir, close_fig = True,
                                                make_fig = make_fig,
                                                show_ind=show_ind,
                                                tags = c_tags,
                                                split_events = split_events,
                                                events_colors=events_colors)
        rec_df = build_recording_dataframe(rec, snippets)
        l_rec_quantif_df.append(c_df)

        dataset.append(rec_df)
    dataset_df = pd.concat(dataset)
    full_rec_df = pd.concat(l_rec_quantif_df)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dset_path = save_dir / 'dataset.h5'
        save_transient_path = save_dir / 'transient_quantification.xlsx'
        dataset_df.to_hdf(save_dset_path.as_posix(), "photometry")
        dataset_df.to_excel(save_dset_path.with_suffix('.xlsx').as_posix())
        full_rec_df.to_excel(save_transient_path.as_posix())


    return dataset_df, full_rec_df


def extract_all_snippets(rec, t_left=2, t_right=5):
    """
    Cut all snippets of signal around all events for a given recording

    Parameters
    ----------
    rec: dict
        Dictionary with all recording data and metadata.
        As returned from load_recording
    t_left: float
        Number of seconds to cut before the event onset
    t_right: float
        Number of seconds to cut after the event onset.

    Returns
    -------
    snippets: dict
        The keys are the same as in the event dictionary
        Then they are split by state (on, off)
        and finally by recording site
    """
    t = rec['time']
    snippets = {}
    for event_name, events in rec['events'].items():
        snippets[event_name] = {}
        for state, events_ts in events.items():
            snippets[event_name][state] = {}
            for site_name, signals in rec['dff'].items():
                snippets[event_name][state][site_name] = {}
                for sensor_name, dff in signals.items():
                    tv, snips, sel_events = get_snippets(t, dff, events_ts, t_left, t_right)
                    if snips.shape[0] == 0:
                        continue
                    snippets[event_name][state][site_name][sensor_name] = {'snips': snips, 't': tv,
                                                                           'event_ts': sel_events}
    return snippets


def quantify_snippets(t, aligned_peth, dur_auc_s):
    auc_bsl, auc_resp = auc_snips(t, aligned_peth, bsl_win=(-dur_auc_s, 0), resp_win=(0, dur_auc_s))
    is_resp = np.logical_and(t > 0, t < dur_auc_s)
    resp = aligned_peth[:, is_resp]
    peak_resp = resp.max(1)
    return {'auc_bsl': auc_bsl, 'auc_resp': auc_resp, 'peak_resp': peak_resp}


def build_recording_dataframe(recording, snippets, dur_auc_s=2):
    rec_dict = {'snip': [], 'site': [], 'event': [], 'state': [], 'trial': [], 't_snip': [],
                'event_ts': []}
    t_snip = None
    for ev_name, all_snips in snippets.items():
        for state, sites_snips in all_snips.items():
            for site, sensors in sites_snips.items():
                for sensor_name, snip_data in sensors.items():
                    t_snip = snip_data['t']
                    z_snips = zscore_snippets(t_snip, snip_data['snips'])
                    n_snips = z_snips.shape[0]
                    rec_dict['sensor'] = sensor_name
                    rec_dict['snip'].extend(list(z_snips))
                    rec_dict['event_ts'].extend(list(snip_data['event_ts']))
                    rec_dict['t_snip'].extend([t_snip]* n_snips)
                    rec_dict['trial'].extend(list(range(n_snips)))
                    rec_dict['site'].extend([site] * n_snips)
                    rec_dict['state'].extend([state] * n_snips)
                    rec_dict['event'].extend([ev_name] * n_snips)
                    # TODO: Think about how and when the quantif should be done (before / after animal averaging)
                    snips_meas = quantify_snippets(t_snip, z_snips, dur_auc_s=dur_auc_s)
                    for k, v in snips_meas.items():
                        prev_meas = rec_dict.setdefault(k, [])
                        prev_meas.extend(v)
                        rec_dict[k] = prev_meas

    rec_df = pd.DataFrame(rec_dict)
    for key, value in recording['infos'].items():
        if isinstance(value, dict):
            continue
        rec_df[key] = value
    return rec_df



if __name__ == '__main__' and True:
    from utilities.settings import upaths
    bp = upaths['J60_PC']
    # bp = upaths ['datapath']
    # datapath = bp / Path('box-AA1_mouse-250005215_strain-pvcre\drug-basal_tai-0\2025_05_27-10_00_41\Fluorescence.csv"Fluorescence.csv')
    # save_path = bp / 'Analysis_filtered'
    # figures_path = save_path / 'Figures'
    # save_path.mkdir(exist_ok=True)
    # figures_path.mkdir(exist_ok=True)
    # dset, df = load_dataset(bp)

    save_fig_dir = bp / 'figures/signal'
    dset, df = load_dataset(bp, make_fig=True, save_dir=save_fig_dir,
                            rec_sites={1: "dHPC", 2: "vHPC"}, events_colors={"Input1": '#FFFFFF00'})
