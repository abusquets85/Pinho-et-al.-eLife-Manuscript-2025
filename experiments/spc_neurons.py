import seaborn as sns
from typing import Optional, List
import re
from pathlib import Path
from pims import Video
from tqdm import tqdm
import numpy as np
from photometry import analysis as pa
from utilities.settings import upaths
from behavior.figures import stars
from utilities.tools import zscore_snippets
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pandas as pd
from behavior.imetronic import Imetronic
from scipy.stats import wilcoxon
import statsmodels.api as sm
from statsmodels.formula.api import ols
from photometry.plotting import plot_signals


SUBTASKS = {'probe test new', 'probe test old', 'habituation', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'ptl', 'ptt',
            'familiarization', 'test (+24h)', 'training1', 'training2'}

TIME_LABEL = 'Time (s)'
DELTA_F_F = r'$\frac{\Delta\ F}{F}$'

CH_SITE = {'dHPC': 1, 'vHPC': 2}
REC_SITE = {1: 'dHPC', 2: 'vHPC'}
BEH_COL = {'Left object': '#006e9055', 'Right object': '#f18f0155',
           'novel': "#BCD97955", 'familiar': "#533B4D55", 'rearing': "#CF637755", 'grooming': "#98728455"}
SITE_COL = {'dHPC': '#427665', 'vHPC': '#6C96DA',
            ('dHPC', 'GCaMP'): '#4DA167', ('dHPC', 'RCaMP'): '#DE3C4B',
            ('vHPC', 'GCaMP'): '#3C7C50', ('vHPC', 'RCaMP'): '#C02130'}


GENO_COL = {'WT': '#008080', 'APP': '#CD5C5C'}
STATES = {0: 'Start', 1: 'Stop'}


def video_info(video_path):
    v = Video(video_path)
    frame_rate = int(v.frame_rate)
    n_frames = len(v)
    frame = v[int(90 * v.frame_rate)]  # One frame after 90 s
    v.close()
    v_infos = {'n_frames': n_frames, 'framerate': frame_rate, 'frame': frame,
               'video_path': video_path}
    return v_infos


def get_task(data_path):
    data_path = Path(data_path)
    parts = data_path.parts
    task = ''
    for p in parts:
        lower_part = p.lower()
        if 'task' in lower_part:
            task = p.replace(' ', '')
    subtask_found = None
    for p in parts:
        if subtask_found is not None:
            break
        lower_part = p.lower()
        for subtask in SUBTASKS:
            if subtask in lower_part:
                subtask_found = subtask
    return task, subtask_found

def parse_path_j60(data_path, with_video=False):
    fluorescence_path = Path(data_path)
    name_session = fluorescence_path.parent.parent.name
    subtask = fluorescence_path.parent.parent.parent.name
    try:
        info = dict([item.split('-') for item in name_session.split('_')])
        info['subtask'] = subtask
        info['filepath'] = fluorescence_path.as_posix()
    except ValueError:
        info = {}


    return info


def parse_path(data_path, with_video=False):
    """
    Extract information from the path to the data file

    Parameters
    ----------
    data_path: str or Path
        Path to the photometry file
    with_video: bool
        Should we look for and open a related video file

    Returns
    -------
    rec_infos: dict
        Information about the recording. Contains:
        date, time, filepath, video (dict if video is loaded), mouse, box, task,
         subtask, sex and genotype
    """

    data_path = Path(data_path)
    date_time = data_path.parent.name
    date, time = date_time.split('-')
    mouse_box = data_path.parent.parent.name
    name_box_split = re.split(r'[\s|_]', mouse_box)
    if len(name_box_split) == 2:
        box_id, mouse_id = name_box_split
    else:
        mouse_id = name_box_split[0]
        box_id = 'X0'
    if not mouse_id.isnumeric():
        mouse_id, box_id = box_id, mouse_id
    l_videos = list(data_path.parent.glob('*.mp4'))
    if len(l_videos) > 0 and with_video:
        video_path = l_videos[0]
        v_infos = video_info(video_path)
    else:
        v_infos = {}
    task, subtask = get_task(data_path)
    sex = 'female' if 'females' in data_path.as_posix().lower() else 'male'

    rec_infos = {'date': date, 'time': time, 'filepath': data_path,
                 'video': v_infos, 'mouse': str(mouse_id), 'box': box_id.upper(),
                 'task': task, 'subtask': subtask, 'sex': sex}
    return rec_infos


def match_pm_beh(rec_infos: dict, beh_path: Path):
    """
    Check if a recording info dictionary (as returned from parse_path) matches a given
    behavioral file path (are they from the same recording). This way we know if a photometry file
    and a behavior file belong to the same trial.

    Parameters
    ----------
    rec_infos: dict
        As returned from parse_path
    beh_path: Path

    Returns
    -------
    bool
    """
    s_mouse = str(rec_infos['mouse'])
    sub_task = rec_infos['subtask']
    # Special case for J60_PC experiments
    if 'J60' in str(rec_infos['filepath']):
        box_id = rec_infos['box'].lower()
        if f'box {box_id}' in beh_path.stem and sub_task.upper() in beh_path.stem:
            return True
    if s_mouse not in beh_path.stem:
        return False
    for p in beh_path.parts:
        lp = p.lower()
        if sub_task in lp:
            return True


def find_behavior(ph_path: Path, rec_infos: dict):
    """
    Find behavioral data files (poly or dlc)
    Available only for Task 3 and 4
    
    Parameters
    ----------
    ph_path: Path
        Path to the photometry data
    rec_infos: dinc
        Dictionary as returned from parse_path to match recordings and behavior

    Returns
    -------
    dlc_file: Path, optional
        Path to the Deeplabcut csv file if found. None otherwise
    poly_file: Path, optional
        Path to the poly .dat file, if found. None otherwise

    """
    beh_base_path_parts = []
    for p in ph_path.parts:
        beh_base_path_parts.append(p)
        if 'SPC' in p or 'CFC' in p:
            break
    beh_path = Path('/'.join(beh_base_path_parts))
    if beh_path == ph_path:
        return None, None
    dlc_file = None
    poly_file = None
    # Look for DLC files
    for dlc_path in beh_path.rglob('*resnet*.csv'):
        if match_pm_beh(rec_infos, dlc_path):
            dlc_file = dlc_path
            break
    # Look for POLY file
    for poly_path in beh_path.rglob('*.dat'):
        if match_pm_beh(rec_infos, poly_path):
            poly_file = poly_path
            break
    return dlc_file, poly_file


def load_recording(file_path, with_video=False, drop_pm_events=False, parse_func=parse_path):
    """
    Load a recording, ie all data related to a given trial.

    Parameters
    ----------
    file_path: str or Path
        Path to photometry data
    with_video: bool
        Should we load the video as well to get some info about it.
        Quite a bit slow
        Defaults to False
    drop_pm_events: bool
        If we find events from Imetronics, should we drop the ones from the photometry file?
        Defaults to False

    Returns
    -------
    recording: dict
        Contains:
        dff: DeltaF/F
        time: time vector
        events: timestamps of events (from the photometry system)
        infos: dict as returned by parse_path augmented with the "behavior" key that contains
               paths to the DLC and poly files.
    """
    file_path = Path(file_path)
    infos = parse_func(file_path, with_video)
    data = pa.analyze_mouse(file_path, infos['mouse'], REC_SITE, trim_duration=90)
    # Synchronization using the start of Imetronic recoded in the RWD system
    start_ttl = data['markers'].get('Input1', {}).get(0, 0)

    # Simplifying the datastructure because just one color but two sites
    # dffs = {site: d['dff_GCaMP'] for site, d in data['channels'].items()}
    beh_files = find_behavior(file_path, infos)
    infos['behavior'] = {'dlc': beh_files[0], 'poly': beh_files[1]}
    if beh_files[1] is not None:
        ev = Imetronic(beh_files[1])
        b_ev = {name: {1 - state: (np.array(ts) / 1000) + start_ttl for state, ts in c_ev.items()}
                for name, c_ev in ev.ts_by_id.items()
                if name in {'LED1', "shock", 'tone'}}
        if drop_pm_events:
            data['markers'] = {}
        data['markers'].update(b_ev)
    recording = {'channels': data['channels'], 'time': data['time'], 'events': data['markers'],
                 'infos': infos}
    return recording


def extract_all_snippets(recording, t_left=30, t_right=120):
    """
    Cut all snippets of signal around all events for a given recording

    Parameters
    ----------
    recording: dict
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
    t = recording['time']
    snippets = {}
    for event_name, events in recording['events'].items():
        snippets[event_name] = {}
        for state, events_ts in events.items():
            snippets[event_name][state] = {}
            for site_name, signals in recording['channels'].items():
                snippets[event_name][state][site_name] = {}
                for wvl, signal in signals.items():
                    tv, c_snips, _ = pa.get_snippets(t, signal, events_ts, t_left, t_right)
                    if c_snips.shape[0] == 0:
                        continue
                    snippets[event_name][state][site_name][wvl] = {'snips': c_snips, 't': tv}
    return snippets


def plot_signal(t, sig, label='dHPC', c='k', ax=None):
    """
    Plot a signal (DF/F/) either in a standalone figure or in a given axes

    Parameters
    ----------
    t: np.ndarray
        Time vector
    sig: np.ndarray
        DF/F
    label: str
        Label for the legend
    c: color spec
    ax: Optional, Matplotlib axes

    Returns
    -------
    ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
    ax.plot(t, sig, c=c, label=label)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(DELTA_F_F)
    return ax


def plot_events(events, c=(.3, .3, .3, .5), event_name='', ax=None):
    """
    Plot events as vertical bars
    Useful to examine the behavior in relation with the photoemtry recording

    Parameters
    ----------
    events: dict or list
        If dict, keys must be 0 and 1.
        0 contains event starts, 1 events stops. This matches the data format from the photometry files
        as well as the way we read them

    c: color spec
    event_name: str
        Name of the event, for the legend
    ax:  Optional, Matplotlib axes

    Returns
    -------
    ax:  Matplotlib axes

    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.set_xlabel('Time (s)')

    labeled = False
    for start, stop in zip(events[0], events[1]):
        if not labeled and event_name != '':
            ax.axvspan(start, stop, ymin=.1, ymax=.9, fc=c, ec='none', label=event_name)
            labeled = True
        else:
            ax.axvspan(start, stop, ymin=.1, ymax=.9, fc=c, ec='none')

    return ax


def plot_peth(snippets: dict, state=0, site_name=None, axs=None,
              title=None, label=None, color=None, scale_ind=False):
    """
    Plot a PETH given some snippets

    Parameters
    ----------
    snippets: dict
        As returned from extract_all_snippets
    state: int
        0 or 1. Depends on if we look at the start (0) or stop (1) of the event
    site_name: Optional, str
        Name of the recording site of interest if not given, all are used
    axs: Optional, array of Matplotlib axes
        Needs to have as many rows as events (len(snippets)) and just one column
    title: Optional, str
        Title to give to the figure
    label: Optional, str
        Label for the plotted curve. If left to None, defaults to the recording site name
    color: Optional
        Color of the curve. If left to None, default to the color associated with the recording site
    scale_ind: Optional, bool
        Should the scale be based on individual curves or the average (default behavior)
        Defaults to False

    """
    if len(snippets) == 0:
        return
    if axs is None:
        n_events = len(snippets)
        fig, axs = plt.subplots(n_events, 1, sharex='all', sharey='all')
        if n_events == 1:
            axs = np.array([axs])
        fig.set_tight_layout(True)
    else:
        fig = axs.reshape(-1)[0].figure
        n_events = axs.shape[0]
    d_ax = dict(zip(snippets.keys(), axs))
    for event_name, data in snippets.items():
        try:
            ax = d_ax[event_name]
        except KeyError:
            print(n_events)
            ax = fig.add_subplot(n_events + 1, 1, n_events + 1)
            d_ax[event_name] = ax
            for ix, fax in enumerate(axs):
                fax.change_geometry(n_events + 1, 1, ix + 1)
            axs = np.hstack((axs, ax))
            n_events = axs.shape[0]
        ax.set_xlabel('Time from event (s)')
        ax.set_ylabel(DELTA_F_F)
        ymin, ymax = ax.get_ylim()
        for site, dff in data[state].items():
            if site_name is not None and site != site_name:
                continue
            col = SITE_COL[site] if color is None else color
            ax.plot(dff['t'], dff['snips'].T, c=col, lw=.3, alpha=.3)
            avg = dff['snips'].mean(0)
            lbl = label if label is not None else site
            ax.plot(dff['t'], avg, c=col, lw=2, label=lbl)
            if scale_ind:
                ymin, ymax = min(ymin, np.min(dff['snips'])), max(ymax, np.max(dff['snips']))
            else:
                ymin, ymax = min(ymin, min(avg)), max(ymax, max(avg))
        ax.set_ylim((ymin, ymax))
        ax.set_title(f'{event_name} - {STATES[state]}')
        ax.legend(facecolor='w', frameon=True, loc='upper right')
    if title is not None:
        fig.suptitle(title)
    return axs


def plot_rec(rec, snippets, save_dir=None):
    """
    Makes the full plot with the recording and its events, for all recording sites as well as all PETH
    for all events, and all recording sites.

    Parameters
    ----------
    rec: dict
        As returned from load_recording
    snippets: dict
        As returned from extract_all_snippets
    save_dir: Optional, str or Path
        Path to save the figures. If None, then don't save

    Returns
    -------
    fig: Matplotlib figure
    """
    # Figure setup
    fig = plt.figure(constrained_layout=True, figsize=(23, 14))
    gs = fig.add_gridspec(2, 2, figure=fig)
    # Plotting raw signals
    n_sites = len(rec['channels'])
    gs_dff = gs[0, :2].subgridspec(n_sites, 1)
    axs_sig = [fig.add_subplot(c_gs) for c_gs in gs_dff]
    plot_signals(rec, axs_sig)
    # Plotting PETHs
    if len(snippets) > 0:
        first_peth = None
        for c_state in range(2):
            gs_peth = gs[1, c_state].subgridspec(len(snippets.keys()), 1)
            axs_peth = np.array([fig.add_subplot(c_gs) for c_gs in gs_peth])
            if first_peth is None:
                first_peth = axs_peth[0]
            for ax in axs_peth:
                ax.sharex(first_peth)
                ax.sharey(first_peth)
            plot_peth(snippets, c_state, axs=axs_peth)

    # Detailed informative title
    info_keys = ('mouse', 'task', 'subtask', 'date', 'time', 'sex', 'genotype')
    infos = rec['infos']
    l_info = [f"{k}: {infos[k]}" for k in info_keys]
    info_str = f'{l_info[0]} {l_info[1]} {l_info[2]}\n{l_info[3]} {l_info[4]}\n{l_info[5]} {l_info[6]}'
    fig.suptitle(info_str, fontsize='large')
    if save_dir is not None:
        save_dir = Path(save_dir)
        fig_name = f"{infos['task']}_{infos['subtask']}_{infos['mouse']}_{infos['date']}_{infos['time']}.png"
        fig.savefig(save_dir / fig_name)
    return fig


def flatten_dict_dataframe(raw_dict, levels, keys, flatten_dict=None, all_res=None):
    if flatten_dict is None:
        flatten_dict = {}
        all_res = []
    c_level = levels[0]
    for c_key, c_data in raw_dict.items():
        flatten_dict[c_level] = c_key
        if len(levels) > 1:
            # Recursively explore this one
            all_res = flatten_dict_dataframe(c_data, levels[1:], keys, flatten_dict, all_res)
        else:
            for k in keys:
                flatten_dict[k] = c_data[k]
            all_res.append(flatten_dict.copy())
    return all_res


def analyse_rec(filepath, t_left=20, t_right=120, save_dir=None, close_fig=False, make_fig=False,
                tags=None, drop_pm_events=False, bef_aft_events=None, split_events=False,
                parse_func=parse_path):
    """
    Entry point to analyse one photometry recording.
    It loads a photometry recording

    Parameters
    ----------
    filepath: str or Path

    t_left: float
        Time in seconds to extract before an event when building a PETH
    t_right: float
        Time in seconds to extract after an event when building a PETH
    save_dir: Optional, Path or str
    close_fig: bool
        Shall we close the figure? Useful when plotting lots of data
    make_fig: bool
        Should we produce the figures or just load the data
    tags: Optional, list of str
        List of strings ('good', 'bad') qualifying the signals.
        Bad signals are dropped
        If not provided (is None) then nothing is dropped
        Defaults to None.
    drop_pm_events: bool
        If we find events from Imetronics, should we drop the ones from the photometry file?
        Defaults to False
    bef_aft_events: Optional, dict
        Keys: Name of events to be used to restrict recording characterization
        (ie number of transients, their frequency, AUC etc...)
        Values: Tuple of float with duration before and duration after, in seconds
    split_events: bool
        Should every presentation of one event be treated as a different event?
        Default to False
    parse_func: callable
        Function used to parse the path of the photometry file to extract metadata

    Returns
    -------
    recording:
        As returned from load_recording.
    snippets:
        As returned from extract_all_snippets
    results:
        As returned from pa.characterize_recording

    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    recording = load_recording(filepath, drop_pm_events=drop_pm_events, parse_func=parse_func)
    if tags is not None:
        recording['channels'] = {site: data
                            for site, data in recording['channels'].items()
                            if tags[CH_SITE[site]-1] == 'good'}
    if split_events:
        recording = split_stimuli(recording)

    snippets = extract_all_snippets(recording, t_left, t_right)

    dt = np.median(np.diff(recording['time']))
    results = {'recording': {}}
    # Ability to split the recording characterization around some events
    if bef_aft_events is None:
        bef_aft_events = {}
    for ev, durations in bef_aft_events.items():
        if ev not in recording['events']:
            continue
        results[ev] = {}
        n_events, _, start_ix, stop_ix = pa.get_start_stop_ix_snippets(recording['time'],
                                                                                recording['events'][ev][0],
                                                                                t_left, t_right)
        if n_events is None:
            continue
        n_before = int(durations[0] / dt)
        n_after = int(durations[1] / dt)
        mask_bef = np.zeros(len(recording['time']), dtype=bool)
        mask_after = np.zeros(len(recording['time']), dtype=bool)
        # Before the event
        for b in start_ix:
            mask_bef[max(0, b-n_before):b] = True
        # After the event
        for e in stop_ix:
            mask_after[e:min(e+n_after, len(mask_after))] = True

        results[ev]['before'] = {site:
                       {wvl: pa.characterize_recording(recording['time'][mask_bef], sig[mask_bef],
                                                       height_th=None, k_mad=3)
                        for wvl, sig in channels.items()}
                    for site, channels in recording['channels'].items()}
        results[ev]['after'] = {site:
                       {wvl: pa.characterize_recording(recording['time'][mask_after], sig[mask_after],
                                                       height_th=None, k_mad=3)
                        for wvl, sig in channels.items()}
                    for site, channels in recording['channels'].items()}
    res_recording = {site:
                         {wvl: pa.characterize_recording(recording['time'], sig,
                                                         height_th=None, k_mad=3)
                          for wvl, sig in channels.items()}
                     for site, channels in recording['channels'].items()}
    results['recording']['all'] = res_recording
    l_df = flatten_dict_dataframe(results, levels=('event', 'period', 'site', 'sensor'),
                                  keys=('frequency', 'amplitude', 'width', 'AUC'))
    df = pd.DataFrame(l_df)
    if make_fig:
        fig = plot_rec(recording, snippets, save_dir)
        if close_fig:
            plt.close(fig)

    df['mouse'] = recording['infos']['mouse']
    df['task'] = recording['infos']['task']
    df['subtask'] = recording['infos']['subtask']
    df['sex'] = recording['infos'].get('sex', 'unknown')

    if save_dir is not None:
        df.to_csv(save_dir / f"{recording['infos']['mouse']}_{recording['infos']['subtask']}.csv",
                  index=False)
    return recording, snippets, res_recording


def load_tags(datapath, tag_rel_path):
    datapath = Path(datapath)
    tags = {}
    with open(tag_rel_path, 'r') as tf:
        for line in tf:
            content = line.strip().split(',')
            abs_path = datapath / content[0]
            tags[str(abs_path)] = content[1:]
    return tags


def analyse_all_recs(datapath, t_left=20, t_right=120, save_dir=None, make_fig=False,
                     drop_bad=True, drop_pm_events=False, bef_aft_events=None, split_events=False,
                     parse_func=parse_path):
    """
    Analyze all recordings in a folder and its subfolders. It is looking for Fluorescence.csv files

    Parameters
    ----------
    datapath: str or Path
        Path to the directory containing data
    t_left: float
        Time in seconds to extract before an event when building a PETH
    t_right: float
        Time in seconds to extract after an event when building a PETH
    save_dir: Optional, Path or str
        Path to save the data
    make_fig: bool
        Should we produce the figures or just load the data
    drop_bad: Optional, bool
        Should recordings that are tagged 'bad' be dropped from the dataset?
        Defaults to True
    drop_pm_events: bool
        If we find events from Imetronics, should we drop the ones from the photometry file?
        Defaults to False
    bef_aft_events: Optional, dict
        Keys: Name of events to be used to restrict recording characterization
        (ie number of transients, their frequency, AUC etc...)
        Values: Tuple of float with duration before and duration after, in seconds
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
    """
    l_files = list(datapath.rglob('Fluorescence.csv'))
    tag_rel_path = datapath / 'tags_rel.csv'
    if tag_rel_path.exists():
        tags = load_tags(datapath, tag_rel_path)
    else:
        tags = {}
    dataset = []
    for c_fp in tqdm(l_files):
        c_tags = tags.get(str(c_fp), None)
        if not drop_bad:
            c_tags = None
        recording, snippets, results = analyse_rec(c_fp, t_left, t_right, save_dir, close_fig=True,
                                                   make_fig=make_fig,
                                                   tags=c_tags, drop_pm_events=drop_pm_events,
                                                   bef_aft_events=bef_aft_events,
                                                   split_events=split_events,
                                                   parse_func=parse_func)
        recording['snippets'] = snippets
        recording['transients'] = results
        recording = average_snippets(recording)
        recording['tags'] = c_tags
        dataset.append(recording)
    return dataset


def common_time_peth(group_data: list, all_peth=None):
    """
    Realign all time vectors

    Parameters
    ----------
    group_data
    all_peth: Optional, dict

    Returns
    -------
    all_peth: dict
    """
    if all_peth is None:
        all_peth = {}
    for key, value in group_data.items():
        if 't' in value.keys() and 'avg' in value.keys():
            all_ts = value['t']
            all_avgs = value['avg']
            t_ref, aligned_peth = pa.align_fragments(all_ts, all_avgs)
            all_peth[key] = {'t': t_ref, 'snips': aligned_peth, 'avg': aligned_peth.mean(0)}
        else:
            all_peth[key] = {}
            common_time_peth(value, all_peth[key])

    return all_peth

def average_snippets(recording):
    recording['peth'] = {}
    for event, all_snips in recording['snippets'].items():
        recording['peth'][event] = {}
        for state, snippets in all_snips.items():
            recording['peth'][event][state] = {}
            for site, s_snips in snippets.items():
                recording['peth'][event][state][site] = {}
                for wvl, c_snip in s_snips.items():
                    avg = np.mean(c_snip['snips'], 0)
                    recording['peth'][event][state][site][wvl] = {'t': c_snip['t'], 'avg': avg}
    return recording


def merge_individual_results(data_path, filename='NOR_quantif'):
    """
    Relaod all CSV files previously saved and merge them into one dataframe and eventually save it

    Parameters
    ----------
    data_path: str or Path
        Path to the folder containing all the CSV to be merged
    filename: Optional, str
        Name of the file to be saved. If None, then don't save the dataframe

    Returns
    -------
    df: pd.DataFrame

    """
    l_df = []
    for csv_file in data_path.glob('*.csv'):
        l_df.append(pd.read_csv(csv_file))
    df = pd.concat(l_df)
    if filename is not None:
        df.to_csv(data_path.parent / f'{filename}.csv', index=False)
    return df


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


def make_recording_frame(recording):
    """
    Produces two DataFrames from a recording, the dictionary produced by analyze_rec
    First DataFrame is relative to compelte DF/F recordings, one per site and sensor
    The Second DataFrame is relative to the behavioral events and their snippets

    Parameters
    ----------
    recording : dict
        As returned by analyze_rec

    Returns
    -------
    ph_df: DataFrame
    ev_df: DataFrame
    """
    infos = {key: value for key, value in recording['infos'].items() if isinstance(value, str)}
    infos['ph_path'] = str(recording['infos']['filepath'])
    infos['dlc_path'] = str(recording['infos']['behavior']['dlc'])
    infos['poly_path'] = str(recording['infos']['behavior']['poly'])
    transients_keys = ('pk', 'frequency', 'amplitude', 'width', 'AUC')
    # We make one dataframe with all the photometry traces and their derived data
    ph_data = []
    for site, channels in recording['channels'].items():
        for sensor, dff in channels.items():
            l_sensors = sensor.split('_')
            sensor_name = l_sensors[0] if len(l_sensors) == 1 else l_sensors[1]
            dff_dict = {'site': site, 'sensor': sensor_name, 'dff': dff,
                     'time': recording['time']}
            c_transient_dict = recording['transients'][site][sensor]
            c_transient_data = {key: value for key, value in c_transient_dict.items()
                                if key in transients_keys}
            dff_dict.update(infos)
            dff_dict.update(c_transient_data)
            ph_data.append(dff_dict)

    # We make one dataframe for all of the events / snippets
    ev_data = []
    for event_name, event_data in recording['events'].items():
        for state, ts in event_data.items():
            event_dict = {'event': event_name, 'state': state, 'timestamp': ts}
            event_dict.update(infos)
            c_snippets = recording['snippets'][event_name][state]
            for site, channels in c_snippets.items():
                for sensor, dff in channels.items():
                    c_snip = dff['snips']
                    c_time = dff['t']
                    z_snip = zscore_snippets(c_time, c_snip)
                    auc_bsl, auc_resp = pa.auc_snips(c_time, c_snip, bsl_win=(-1, 0), resp_win=(0, 1))
                    l_sensors = sensor.split('_')
                    sensor_name = l_sensors[0] if len(l_sensors) == 1 else l_sensors[1]
                    event_dict.update({'snip': c_snip, 'z_snips': z_snip, 'peth': z_snip.mean(0),
                                      'time': c_time, 'sensor': sensor_name, 'site': site,
                                       'auc_snip_bsl': auc_bsl, 'auc_snip_resp': auc_resp})
                    ev_data.append(event_dict.copy())

    return pd.DataFrame(ph_data), pd.DataFrame(ev_data)


def make_dataset_frame(dataset, dur_auc_s=2):
    """
    Produces the DataFrames for all recordings in a dataset

    Parameters
    ----------
    dataset : list of dict
        As returned by analyse_all_recs
    dur_auc_s: float
        Duration over which the AUC of the PETH is computed for quantification

    Returns
    -------

    """
    all_ph_df = []
    all_ev_df = []
    for recording in dataset:
        c_ph_df, c_ev_df = make_recording_frame(recording)
        all_ph_df.append(c_ph_df)
        all_ev_df.append(c_ev_df)

    events_df = pd.concat(all_ev_df).reset_index()
    # We realign all PETH of all recording on the same time vector
    t, al_peth = pa.align_fragments(events_df['time'].to_numpy(),
                                    events_df['peth'].to_numpy())

    auc_bsl, auc_resp = pa.auc_snips(t, al_peth, bsl_win=(-dur_auc_s, 0), resp_win=(0, dur_auc_s))
    is_resp = np.logical_and(t > 0, t < dur_auc_s)
    resp = al_peth[:, is_resp]
    peak_resp = resp.max(1)
    # and save the re-aligned PETH and time to the DataFrame
    events_df['time'] = [t for _ in al_peth]
    events_df['peth'] = [p for p in al_peth]
    events_df['auc_bsl'] = auc_bsl
    events_df['auc_resp'] = auc_resp
    events_df['peak_resp'] = peak_resp
    return pd.concat(all_ph_df), events_df


def select_from_dataset(dataset_df, filter_dict):
    """
    Select only a subset of the available PETH based on criteria passed as a dictionary


    Parameters
    ----------
    dataset_df : DataFrame
        As returned by make_dataset_frame
    filter_dict : dict
        {column_name: selected_value}
        For example: {"site": "dHPC", "sensor": "GCaMP",
                      "animal": ['230049345', '230049347', '230049351', '230049349']}

    Returns
    -------
    selected_df: DataFrame
        Subset of event_df
    """
    mask = np.ones(len(dataset_df), dtype=bool)
    for column_name, value in filter_dict.items():
        if isinstance(value, list) or isinstance(value, tuple):
            new_mask = dataset_df[column_name].isin(value)
        else:
            new_mask = dataset_df[column_name] == value
        mask = mask * new_mask
    selected_df = dataset_df[mask]
    return selected_df


def all_peth_by_animal(event_df, subtask, event_name, figures_path: Path):
    animals = event_df["mouse"].unique().tolist()
    for c_animal in animals:
        fig, axs = all_peth_selected(event_df, subtask, event_name, c_animal)
        save_path = figures_path / f'by_animal/{subtask}/{event_name}'
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f'{c_animal}.png')
        plt.close(fig)


def all_peth_selected(event_df, subtask, event_name, animals=None, show_ind=True,
                      show_heatmap=False):
    if animals is None:
        animals = event_df["mouse"].unique().tolist()
    selected = select_from_dataset(event_df, {'subtask': subtask, 'event': event_name})
    all_sensors = np.sort(selected["sensor"].unique())
    all_sites = np.sort(selected["site"].unique())
    fig, axs = plt.subplots(len(all_sensors), len(all_sites), squeeze=False, figsize=(10, 7),
                            sharex='all', sharey='all')
    for ix_site, c_site in enumerate(all_sites):
        axs[0, ix_site].set_title(c_site)
        all_sensors_selected(all_sensors, animals, axs, c_site, ix_site, selected, show_ind,
                             show_heatmap=show_heatmap)
    fig.suptitle(f'Subtask: {subtask} - Event: {event_name}')
    axs[0, 0].set_ylim(-2, 6)
    fig.set_tight_layout(True)
    return fig, axs


def average_peth_per_mouse(df):
    gp = df.groupby('mouse')
    avg_peth = []
    for mouse, indices in gp.groups.items():
        snips = df.loc[indices]['peth'].to_numpy()
        avg = snips.mean(0)
        avg_peth.append(avg)
    return np.vstack(avg_peth)


def all_sensors_selected(all_sensors, animals, axs, c_site, ix_site, selected, show_ind,
                         show_stats=False, show_heatmap=False):
    l_diffs_df = []
    for ix_sensor, c_sensor in enumerate(all_sensors):
        c_ax = axs[ix_sensor, ix_site]
        sel_dict = {"sensor": c_sensor, "site": c_site,
                    'mouse': animals, "state": 0}
        c_sel = select_from_dataset(selected, sel_dict)
        # raise ValueError
        if len(c_sel) == 0:
            break
        time = c_sel['time'].iloc[0]
        # peth = np.vstack(c_sel['peth'].to_numpy())
        peth = average_peth_per_mouse(c_sel)
        one_peth(c_ax, c_sensor, c_site, peth, time, show_ind=show_ind, show_heatmap=show_heatmap,
                 heatmap_lbl='Animal')
        pvalue, mouse_diffs = response_paired_stats(c_sel)
        print(f"{c_site} - {c_sensor} - {pvalue}")
        sel_dict['mouse'] = c_sel['mouse'].unique().tolist()
        sel_dict['diffs'] = mouse_diffs
        l_diffs_df.append(pd.DataFrame(sel_dict))
        if show_stats:
            y_loc = np.mean(peth, axis=0)[np.searchsorted(time, 0): np.searchsorted(time, 1)].max()
            c_ax.text(0.5, y_loc+0.8, stars(pvalue), horizontalalignment='center')
            c_ax.plot([0, 1], [y_loc+0.5, y_loc+0.5], color='black')
    if len(l_diffs_df) > 0:
        return pd.concat(l_diffs_df)


def response_paired_stats(event_df):
    animals = event_df["mouse"].unique().tolist()
    mouse_diffs = []
    for mouse in animals:
        mouse_df = select_from_dataset(event_df, {"mouse": mouse})
        auc_snip_bsl = np.hstack(mouse_df['auc_snip_bsl'].to_numpy())
        auc_snip_resp = np.hstack(mouse_df['auc_snip_resp'].to_numpy())
        # resp = auc_snip_resp - auc_snip_bsl

        resp = mouse_df['peak_resp']
        mouse_diffs.append(np.mean(resp))

    w_res = wilcoxon(mouse_diffs)

    return w_res.pvalue, mouse_diffs


def one_peth(c_ax, c_sensor, c_site, peth, time, show_ind=True, show_heatmap=False,
             heatmap_lbl='Animal'):
    avg = peth.mean(0)
    n_trials = peth.shape[0]
    sem = np.std(peth, 0) / np.sqrt(n_trials)
    if show_heatmap:
        divider = make_axes_locatable(c_ax)
        hm_ax = divider.append_axes('top', size='30%', pad='5%')
        # hm_ax.set_title(f'{site_name}', fontsize=16)
        # ax.plot(t_snips, snips.T, c='k', lw=.5, alpha=.5)
        # zpeth = zscore_snippets(time, peth)
        hm_ax.imshow(peth, extent=(time[0], time[-1], 0, n_trials),
                     interpolation='none', aspect='auto', vmin=0, vmax=4)
        hm_ax.spines['bottom'].set_visible(False)
        hm_ax.xaxis.set_ticks_position('none')
        hm_ax.xaxis.set_ticklabels([])
        hm_ax.set_yticks([n_trials], [str(n_trials)])
        hm_ax.set_ylabel(heatmap_lbl)
    if show_ind:
        c_ax.plot(time, peth.T, lw=.5, color='.2')
    else:
        c_ax.fill_between(time, avg-sem, avg+sem, facecolor=SITE_COL[(c_site, c_sensor)],
                          edgecolor='none', alpha=.5)

    c_ax.plot(time, avg, c=SITE_COL[(c_site, c_sensor)], lw=3)
    c_ax.set_xlabel('Time (s)')
    c_ax.set_ylabel(r'$Z-scored\ \frac{\Delta F}{F}$ ' + c_sensor)
    c_ax.figure.align_ylabels()
    c_ax.axvline(0)


def group_tasks(tasks):
    gp_tasks = {}
    for t in tasks:
        task_match = re.match("([a-z]+)", t)
        if task_match is None:
            continue
        task_name = task_match.group(0)
        prev = gp_tasks.get(task_name, [])
        prev.append(t)
        gp_tasks[task_name] = prev
    gp_tasks = {task_name: tasks for task_name, tasks in gp_tasks.items() if len(tasks) > 1}
    return gp_tasks


def filter_animals(event_df, resp_th=0):
    animals = event_df["mouse"].unique().tolist()
    all_sensors = np.sort(event_df["sensor"].unique())
    all_sites = np.sort(event_df["site"].unique())
    l_df = []
    for c_animal in animals:
        for ix_sensor, c_sensor in enumerate(all_sensors):
            for ix_site, c_site in enumerate(all_sites):
                selected = select_from_dataset(event_df,
                                               {'subtask': 'training2', 'event': 'shock',
                                                "state": 0,
                                                'mouse': c_animal,
                                                "sensor": c_sensor, "site": c_site})
                c_filter_df = pd.DataFrame({'mouse': c_animal, 'sensor': c_sensor,
                                            'site': c_site,
                                            'good': selected['auc_resp']> resp_th,
                                            'auc':selected['auc_resp']})
                l_df.append(c_filter_df)
    filter_df = pd.concat(l_df)
    return filter_df


def apply_filter(event_df, filter_df):

    keep_ix = []
    for ix_row, row in filter_df.iterrows():
        filter_dict = row.to_dict()
        if not filter_dict['good']:
            continue
        # event_df.loc[sel_ev.index, 'good'] = filter_dict['good']
        filter_dict_p = {k: v for k, v in filter_dict.items() if k in event_df.columns}
        sel_ev = select_from_dataset(event_df, filter_dict_p)
        keep_ix.extend(sel_ev.index.to_list())
    return event_df.iloc[keep_ix]


def count_sig_peth(event_df, subtask, event_name, animals=None, z_th=2):
    if animals is None:
        animals = event_df["mouse"].unique().tolist()
    selected = select_from_dataset(event_df, {'subtask': subtask, 'event': event_name})
    all_sensors = np.sort(selected["sensor"].unique())
    all_sites = np.sort(selected["site"].unique())
    fig, axs = plt.subplots(len(all_sensors), len(all_sites), squeeze=False, figsize=(10, 7),
                            sharex='all', sharey='all')
    for ix_site, c_site in enumerate(all_sites):
        axs[0, ix_site].set_title(c_site)
        count_sig_all_sensors(all_sensors, animals, axs, c_site, ix_site, selected, z_th)
    fig.suptitle(f'Subtask: {subtask} - Event: {event_name}')
    fig.set_tight_layout(True)
    return fig, axs


def count_sig_all_sensors(all_sensors, animals, axs, c_site, ix_site, selected, z_th):
    for ix_sensor, c_sensor in enumerate(all_sensors):
        c_ax = axs[ix_sensor, ix_site]
        c_sel = select_from_dataset(selected, {"sensor": c_sensor, "site": c_site,
                                               'mouse': animals, "state": 0})
        if len(c_sel) == 0:
            break
        time = c_sel['time'].iloc[0]
        peth = np.vstack(c_sel['peth'].to_numpy())
        is_sig = peth > z_th
        n_sig = np.sum(is_sig, 0)
        c_ax.plot(time, n_sig)
        c_ax.set_xlabel(TIME_LABEL)
        c_ax.set_ylabel(r'Number of trials with a $Z-\frac{\Delta F}{F} >$ ' + f'{z_th}')


def make_figure(event_df, subtask, event_name, save_path=None, z_th=2,
                show_heatmap=False):
    if not isinstance(subtask, list):
        subtask = list(subtask)
    if 'training' in subtask[0]:
        task_name = 'Conditioning'
    elif 'pc' in subtask[0]:
        task_name = 'Preconditioning'
    elif 'ptl' in subtask[0]:
        task_name = 'Probe test light'
    elif 'ptt' in subtask[0]:
        task_name = 'Probe test tone'

    event_names_d = {'LED1': 'Light', 'shock': 'Shock', 'tone': 'Tone'}
    ev_name = event_names_d[event_name]
    title = f'{task_name} - {ev_name}'
    fig, axs = all_peth_selected(event_df, subtask, event_name, None, show_ind=False,
                                 show_heatmap=show_heatmap)
    for c_ax in axs.reshape(-1):
        c_ax.set_xlabel(f'Time from {ev_name.lower()} (s)')
        if event_name == 'shock':
            c_ax.set_ylim(-1, 12)
    fig.suptitle(title)
    fig_c, axs_c = count_sig_peth(event_df, subtask, event_name, None, z_th=z_th)
    fig_c.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path / f'PSTH_{task_name}_{ev_name}.png')
        fig_c.savefig(save_path / f'Sig_{task_name}_{ev_name}.png')


def peth_and_count_sig(event_df, subtask, event_name, site, animals=None, show_ind=True, z_th=2,
                       show_heatmap=False):
    if animals is None:
        animals = event_df["mouse"].unique().tolist()
    selected = select_from_dataset(event_df, {'subtask': subtask, 'event': event_name, 'site': site})
    all_sensors = np.sort(selected["sensor"].unique())
    fig, axs = plt.subplots(len(all_sensors), 2, squeeze=False, figsize=(10, 7),
                            sharex='all', sharey='col')
    diffs_df = all_sensors_selected(all_sensors, animals, axs, site, 0, selected, show_ind,
                                    show_stats=True, show_heatmap=show_heatmap)
    count_sig_all_sensors(all_sensors, animals, axs, site, 1, selected, z_th)
    fig.suptitle(f'Subtask: {subtask} - Event: {event_name} - {site}')
    axs[0, 0].set_ylim(-1.5, 3)
    axs[0, 1].set_ylim(-0.5, 20)
    fig.set_tight_layout(True)
    return fig, axs, diffs_df


def plot_dff_diffs(diff_df):
    task_name = diff_df['task'].unique().tolist()[0]
    event_name = diff_df['event'].unique().tolist()[0]
    sites = np.sort(diff_df['site'].unique())
    sensors = np.sort(diff_df['sensor'].unique())
    fig, axs = plt.subplots(1, len(sites), sharey='all', figsize=(3.7, 4.8))
    print(f'*** {task_name} - {event_name} ***')
    for ix, site in enumerate(sites):
        c_sel = select_from_dataset(diff_df, {"site": site})
        data = [c_sel[c_sel['sensor'] == s]['diffs'] for s in sensors]
        ax = axs[ix]
        bp = ax.boxplot(data, patch_artist=True, labels=sensors, widths=.8)
        for s_ix, c_sensor in enumerate(sensors):
            color_boxplot(bp, c_sensor, s_ix, site)
            pval = wilcoxon(data[s_ix])
            sig = stars(pval.pvalue)
            print(f'{site}, {c_sensor}, p={pval.pvalue}, n={len(data[s_ix])}')
            ax.text(s_ix+1, .995, sig, transform=ax.get_xaxis_transform(), va='top')
        ax.set_ylabel(f'Maximal value of DF/F in a 1s window after {diff_df["event"].iloc[0]}')
        ax.set_title(f'{site}')
    fig.suptitle(f'{diff_df["task"].iloc[0]}')
    fig.set_tight_layout(True)
    return fig


def color_boxplot(box_plots, sensor, box_ix, site):
    box_plots['boxes'][box_ix].set_facecolor(SITE_COL[(site, sensor)])
    box_plots['boxes'][box_ix].set_edgecolor('none')
    box_plots['medians'][box_ix].set_color('white')
    box_plots['fliers'][box_ix].set_marker('.')
    box_plots['fliers'][box_ix].set_markerfacecolor(SITE_COL[(site, sensor)])
    box_plots['fliers'][box_ix].set_markeredgecolor('none')


def make_figure_by_region(event_df, subtask, event_name, save_path=None, show_ind=False, z_th=2,
                          show_heatmap=False):
    if not isinstance(subtask, list):
        subtask = list(subtask)
    if 'training' in subtask[0]:
        task_name = 'Conditioning'
    elif 'pc' in subtask[0]:
        task_name = 'Preconditioning'
    elif 'ptl' in subtask[0]:
        task_name = 'Probe test light'
    elif 'ptt' in subtask[0]:
        task_name = 'Probe test tone'
    else:
        task_name = '_task_'

    event_names_d = {'LED1': 'Light', 'shock': 'Shock', 'tone': 'Tone'}
    ev_name = event_names_d.get(event_name, event_name)
    all_sites = np.sort(event_df["site"].unique())
    l_diffs_df = []
    for site in all_sites:
        title = f'{task_name} - {ev_name} - {site}'
        fig, axs, diffs_df = peth_and_count_sig(event_df, subtask, event_name, site, None,
                                                show_ind=show_ind, z_th=z_th,
                                                show_heatmap=show_heatmap)
        diffs_df['site'] = site
        diffs_df['task'] = task_name
        diffs_df['event'] = event_name

        l_diffs_df.append(diffs_df)
        for c_ax in axs.reshape(-1):
            c_ax.set_xlabel(f'Time from {ev_name.lower()} (s)')
        if event_name == 'shock':
            axs[0, 0].set_ylim(-1, 12)
            for k in np.arange(axs.shape[0]):
                axs[k, 0].text(0, 11, f'{ev_name}', horizontalalignment='center',
                               verticalalignment='center', backgroundcolor='white')
        else:
            for k in np.arange(axs.shape[0]):
                axs[k, 0].text(0, 5, f'{ev_name}', horizontalalignment='center',
                               verticalalignment='center', backgroundcolor='white',
                               transform=axs[k, 0].get_xaxis_transform())

        fig.suptitle(title)

        if save_path is not None:
            fig.savefig(save_path / f'{task_name}_{ev_name}_{site}.png')
            fig.savefig(save_path / f'{task_name}_{ev_name}_{site}.svg')
    if len(l_diffs_df) > 0:
        diffs_df = pd.concat(l_diffs_df)
        fig = plot_dff_diffs(diffs_df)
        if save_path is not None:
            fig.savefig(save_path / f'{task_name}_{ev_name}_{site}_boxplot.png')
            fig.savefig(save_path / f'{task_name}_{ev_name}_{site}_boxplot.svg')
        return diffs_df


def figure_split_subtask(event_df, subtasks, event_name, save_path=None, z_th=2,
                         show_heatmap=False):
    l_diffs_df = []
    plt.ioff()
    for subtask in subtasks:
        c_diffs_df = make_figure_by_region(event_df, [subtask], event_name,
                                           None, False, z_th,
                                           show_heatmap=show_heatmap)
        plt.close('all')
        c_diffs_df['subtask'] = subtask
        l_diffs_df.append(c_diffs_df)
    full_diff_df = pd.concat(l_diffs_df)
    plt.ion()
    sensors = full_diff_df['sensor'].unique()
    sites = full_diff_df['site'].unique()
    n_rows = len(sensors)
    n_cols = len(sites)
    fig, axs = plt.subplots(n_rows, n_cols, sharey='row', sharex='all', figsize=(6, 9), squeeze=False)
    all_subtasks = np.sort(full_diff_df['subtask'].unique())
    xtick_labels = [t.upper() for t in all_subtasks]
    ix_sensor = 0
    for c_sensor in sensors:
        ix_site = 0
        for c_site in sites:
            ax = axs[ix_sensor, ix_site]
            filter_d = {"site": c_site, 'sensor': c_sensor}
            sel_data = select_from_dataset(full_diff_df, filter_d)
            data = [sel_data[sel_data['subtask']==st]['diffs'] for st in all_subtasks]
            l_bps = ax.boxplot(data, patch_artist=True, widths=.8)
            for ix in range(len(all_subtasks)):
                color_boxplot(l_bps, c_sensor, ix, c_site)
            ax.set_title(f'{c_site} - {c_sensor}')
            ax.set_xticks(range(1, 1 + len(xtick_labels)), xtick_labels)
            ax.set_ylabel(f'Maximal value of DF/F in a 1s window after {full_diff_df["event"].iloc[0]}')
            ix_site += 1
        ix_sensor += 1
    fig.set_tight_layout(True)
    if save_path is not None:
        task_name = full_diff_df['task'].iloc[0]
        fig.savefig(save_path / f'{task_name}_{event_name}_boxplots_over_phases.png')
        fig.savefig(save_path / f'{task_name}_{event_name}_boxplots_over_phases.svg')
    return full_diff_df

def figure_split_presentation(event_df, subtasks, events, save_path=None, z_th=2):
    l_diffs_df = []
    plt.ioff()
    for c_event in events:
        c_diffs_df = make_figure_by_region(event_df, subtasks, c_event,
                                           None, False, z_th)
        plt.close('all')
        c_diffs_df['presentation'] = c_event
        l_diffs_df.append(c_diffs_df)
    full_diff_df = pd.concat(l_diffs_df)

    plt.ion()
    sensors = full_diff_df['sensor'].unique()
    sites = full_diff_df['site'].unique()
    n_rows = len(sensors)
    n_cols = len(sites)
    fig, axs = plt.subplots(n_rows, n_cols, sharey='row', sharex='all', figsize=(6, 9))
    all_presentations = np.sort(full_diff_df['presentation'].unique())
    xtick_labels = [f'Light\n#{ix+1}' for ix in range(len(all_presentations))]
    ix_sensor = 0
    for c_sensor in sensors:
        ix_site = 0
        for c_site in sites:
            ax = axs[ix_sensor, ix_site]
            filter_d = {"site": c_site, 'sensor': c_sensor}
            sel_data = select_from_dataset(full_diff_df, filter_d)
            data = [sel_data[sel_data['presentation']==st]['diffs'] for st in all_presentations]
            l_bps = ax.boxplot(data, patch_artist=True, widths=.8)
            for ix in range(len(all_presentations)):
                color_boxplot(l_bps, c_sensor, ix, c_site)
            ax.set_title(f'{c_site} - {c_sensor}')
            ax.set_xticks(range(1, 1+len(xtick_labels)), xtick_labels)
            ax.set_ylabel(f'Maximal value of DF/F in a 1s window after light')
            ix_site += 1
        ix_sensor += 1
    fig.set_tight_layout(True)
    if save_path is not None:
        task_name = full_diff_df['task'].iloc[0]
        fig.savefig(save_path / f'{task_name}_Light_boxplots_over_presentations.png')
        fig.savefig(save_path / f'{task_name}_Light_boxplots_over_presentations.svg')
    return full_diff_df


if __name__ == '__main__' and False:
    fp = Path('/mnt/Data_1/NAS/Data/Busquet/Astros/Task 3- SPC/fiberphotometry/Males/PTT/2300035560_F2/2023_02_17-10_52_01/Fluorescence.csv')
    rec, snips, res = analyse_rec(fp, 30, 30, make_fig=True)

# ashley: use this block
if __name__ == '__main__' and True:   # TRUE
    plt.ion()
    base_path = upaths['spc_neurons']
    fig_path = base_path / 'figures'
    csv_path = base_path / 'quantification_csv'
    csv_path_split = base_path / 'quantification_csv_split'
    csv_path.mkdir(exist_ok=True)
    csv_path_split.mkdir(exist_ok=True)
    dset_all = analyse_all_recs(base_path, t_left=2, t_right=5, drop_pm_events=True,
                                split_events=False, save_dir=csv_path,
                                bef_aft_events={'LED1': (60, 60)},
                                parse_func=parse_path)

    dset = analyse_all_recs(base_path, t_left=2, t_right=5, drop_pm_events=True,
                            split_events=True, save_dir=csv_path_split,
                            bef_aft_events={'LED1': (60, 60)},
                            parse_func=parse_path)

    ph_df_all, ev_df_all = make_dataset_frame(dset_all, dur_auc_s=10)
    ph_df, ev_df = make_dataset_frame(dset, dur_auc_s=10)
    fdf = filter_animals(ev_df_all, 2)
    f_ev_df = apply_filter(ev_df, fdf)
    f_ev_df_all = apply_filter(ev_df_all, fdf)

# ashley: use this block
if __name__ == '__main__' and False:
    paper_fig = fig_path / 'paper'
    paper_fig.mkdir(exist_ok=True)
    make_figure(f_ev_df, ['training1', 'training2'], 'shock',
                save_path=paper_fig)
    make_figure(f_ev_df, ['training1', 'training2'], 'LED1',
                save_path=paper_fig)
    make_figure(f_ev_df, ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'], 'LED1',
            save_path=paper_fig)
    make_figure(f_ev_df, ['ptl'], 'LED1', save_path=paper_fig)
    make_figure(f_ev_df, ['ptt'], 'tone', save_path=paper_fig)

# ashley: use this block for reorganized paper figures
if __name__ == '__main__' and False:   # TRUE
    reorg_fig = fig_path / 'J60_PC'
    reorg_fig.mkdir(exist_ok=True)
    make_figure_by_region(f_ev_df_all, ['training1', 'training2'], 'shock',
                          save_path=reorg_fig, z_th=2, show_heatmap=True)
    make_figure_by_region(f_ev_df_all, ['training1', 'training2'], 'LED1',
                          save_path=reorg_fig, z_th=2, show_heatmap=True)
    make_figure_by_region(f_ev_df_all, ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'], 'LED1',
                          save_path=reorg_fig, z_th=2, show_heatmap=True)
    dd = figure_split_subtask(f_ev_df_all, ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'],
                              'LED1',
                              save_path=upaths['spc_neurons']/'figures/revisions',
                              show_heatmap=True)
    dsh = figure_split_subtask(f_ev_df_all, ['training1', 'training2'],
                              'shock',
                              save_path=upaths['spc_neurons']/'figures/revisions',
                              show_heatmap=True)
    # Make an ANOVA on this dataframe
    dorsal_gcamp = dd[(dd["sensor"]=="GCaMP") & (dd["site"] == "dHPC")]
    dorsal_rcamp = dd[(dd["sensor"]=="RCaMP") & (dd["site"] == "dHPC")]
    ventral_gcamp = dd[(dd["sensor"]=="GCaMP") & (dd["site"] == "vHPC")]
    ventral_rcamp = dd[(dd["sensor"]=="RCaMP") & (dd["site"] == "vHPC")]
    lm = ols('diffs ~ C(subtask)', data=dorsal_gcamp).fit()
    table = sm.stats.anova_lm(lm, typ=2)
    lm = ols('diffs ~ C(subtask)', data=dorsal_rcamp).fit()
    table = sm.stats.anova_lm(lm, typ=2)
    print(table)
    lm = ols('diffs ~ C(subtask)', data=ventral_gcamp).fit()
    table = sm.stats.anova_lm(lm, typ=2)
    print(table)
    lm = ols('diffs ~ C(subtask)', data=ventral_rcamp).fit()
    table = sm.stats.anova_lm(lm, typ=2)
    print(table)
    # make_figure_by_region(f_ev_df, ['ptl'], 'LED1', save_path=reorg_fig, z_th=2)
    # make_figure_by_region(f_ev_df, ['ptt'], 'tone', save_path=reorg_fig, z_th=2)




if __name__ == '__main__' and False:

    split_str = "average_session"
    # c_task = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6']
    # c_task = ['training1'] # , 'training2']
    # c_task = ['pc3', 'pc4', 'pc5', 'pc6']
    c_task = ['ptl']
    event_name = 'LED1'
    # event_name = 'tone'
    # tdf = select_from_dataset(ev_df, {"subtask": c_task})
    fig, _ = all_peth_selected(f_ev_df, c_task, event_name, None, show_ind=False)
    selected = select_from_dataset(f_ev_df,
                                   {'subtask': c_task, 'event': event_name, 'state': 0})
    sns.catplot(selected, x='sensor', y='auc_resp', hue='site', kind='box')

if __name__ == '__main__' and False:

    df = merge_individual_results(csv_path, 'pre_post_pc_led')
    df_led = df[(df['event']=='LED1') & (df['subtask'].isin(['pc1', 'pc2', "pc3", "pc4", "pc5", 'pc6']))]
    df_led.reset_index(inplace=True, drop=True)
    comp_fig_path = fig_path / 'quantification'
    comp_fig_path.mkdir(exist_ok=True)
    for v in ('frequency', 'amplitude', 'width', 'AUC'):
        sns.catplot(df_led, x='sensor', y=v, hue='period', row='site', kind='box', showfliers=False)
        plt.savefig(comp_fig_path / f'{v}.png')

    # Probe test snippets
    # Subtasks
    ptl = (ev_df['subtask'] == 'ptl') & (ev_df['event'] == 'LED1')
    ptt = (ev_df['subtask'] == 'ptt') & (ev_df['event'] == 'tone')
    # Site / Sensor
    dhpc = (ev_df['site'] == 'dHPC')
    vhpc = (ev_df['site'] == 'vHPC')
    exc = (ev_df['sensor'] == 'RCaMP')
    inh = (ev_df['sensor'] == 'GCaMP')
    # Animals
    plt.figure(figsize=[10, 5])
    animals = ('230049345', '230049347', '230049351', '230049349')
    # Figure
    g_animals = ev_df['mouse'].isin(animals)
    exc_d_ptl = ev_df[ptl & dhpc & inh & g_animals]

    peth = np.vstack(exc_d_ptl['peth'].to_numpy())
    time = exc_d_ptl['time'].iloc[0]
    # TODO: Filter by how reactive animals are to the shock
    sel = select_from_dataset(ev_df, {'subtask': 'training2', 'event': 'shock', 'site': 'dHPC', 'sensor': 'GCaMP'})
    sel['responsive'] = sel['auc_resp'] > 5
    # TODO: Figures: All events / all phases split or not

# TODO : Figure for PC (all 6 pooled) + Quantif (number of significant trials)
# TODO : PTL/PTT traces + quantif - supplementary?
# TODO: Training: 8 s of light + shock (2s) + end of light and shock
# TODO: PETH during training: LED1 only (-2s to +5s) and then shock only + quantif

if True:
    save_path = upaths['spc_neurons'] / "figures/published_tables"
    # Figure 2
    df_fig_2 = make_figure_by_region(f_ev_df_all, ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'], 'LED1',
                          save_path=save_path, z_th=2, show_heatmap=True)
    df_fig_2.to_excel(save_path / 'figure_2.xlsx', index=None)

    # Figure 2 Suppl Fig 1 A
    dd = figure_split_subtask(f_ev_df_all, ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'],
                              'LED1',
                              save_path=save_path,
                              show_heatmap=True)
    dd.to_excel(save_path / 'figure_2_suppl_1_A.xlsx', index=None)

    # Figure 2 Suppl Fig 1 B
    df_f2s1b = figure_split_presentation(f_ev_df,
                                         ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'],
                                         ['LED1_0', 'LED1_1', 'LED1_2', 'LED1_3', 'LED1_4'],
                                         save_path=save_path)
    df_f2s1b.to_excel(save_path / 'figure_2_suppl_1_B.xlsx', index=None)

    # Figure 2 Suppl Fig 1 C
    dsh = figure_split_subtask(f_ev_df_all, ['training1', 'training2'],
                          'shock',
                          save_path=save_path,
                          show_heatmap=True)
    dsh.to_excel(save_path / 'figure_2_suppl_1_C.xlsx', index=None)

    # Fig 2 Suppl Fig 2
    df_fig2_s2 = make_figure_by_region(f_ev_df_all, ['training1', 'training2'], 'LED1',
                          save_path=save_path, z_th=2, show_heatmap=True)
    df_fig2_s2.to_excel(save_path / 'figure_2_suppl_2.xlsx', index=None)

    # Fig 2 Suppl Fig 3
    df_fig2_s3 = make_figure_by_region(f_ev_df_all, ['training1', 'training2'],
                                       'shock', z_th=2, show_heatmap=True,
                                       save_path=save_path)
    df_fig2_s3.to_excel(save_path / 'figure_2_suppl_3.xlsx', index=None)

    # Figure 3 Suppl Fig 1 A
    # This one is tricky because it requires a different dataset
    dset_all = analyse_all_recs(upaths['J60_PC'], t_left=2, t_right=5, drop_pm_events=True,
                                split_events=False, save_dir=None,
                                bef_aft_events={'LED1': (60, 60)},
                                parse_func=parse_path_j60)

    dset = analyse_all_recs(upaths['J60_PC'], t_left=2, t_right=5, drop_pm_events=True,
                            split_events=True, save_dir=None,
                            bef_aft_events={'LED1': (60, 60)},
                            parse_func=parse_path_j60)

    ph_df_all, ev_df_all = make_dataset_frame(dset_all, dur_auc_s=10)
    ph_df, ev_df = make_dataset_frame(dset, dur_auc_s=10)

    dd3 = figure_split_subtask(ev_df_all, ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'],
                               'LED1',
                               save_path=save_path / "J60",
                               show_heatmap=True)
    dd3.to_excel(save_path / 'figure_3_E.xlsx', index=None)

    df_fig3_D = make_figure_by_region(ev_df_all, ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6'] ,
                                      'LED1', z_th=2, show_heatmap=True,
                                      save_path=save_path/ "J60")
    df_fig3_D.to_excel(save_path / 'figure_3_D.xlsx', index=None)
