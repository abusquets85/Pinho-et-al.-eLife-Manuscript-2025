"""
Once upon a time in 2022...
Updated on Tue May 13/2025 

@author: remi
@tested: cramon
"""


import warnings
from pathlib import Path
from typing import Sequence, Optional, Dict  # added
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utilities.tools import clean_ax, zscore_snippets, merge_infos
from utilities import colors as c
from utilities.colors import group_colors
# try:
import colorcet

cmap = colorcet.cm.dimgray.reversed()
# except ImportError:
#     cmap = plt.cm.tab20
# cmap = plt.cm.Blues
ev_cmap = plt.cm.Pastel2

TIME_LABEL = 'Time (s)'
DELTA_F_F = r'$\frac{\Delta\ F}{F}$'
EVENT_STATES_NAMES = {0: 'Start', 1: 'Stop'}
plt.ion()


def plot_raw_data(data, ch_num, ch_wvl, session_name='', axs=None):
    """
    Plots the raw data, allowing a first inspection of the signal.
    In green: our raw calcium dependent signal [470nm]
    In purple: the isosbestic signal (control) [410nm]


    Parameters
    ----------
    data : np.ndarray
        Contains our fluorescence data points.
    ch_num: tuple
        The different channel numbers eg. (1, 2)
    ch_wvl: tuple
        The different wavelengths recorded eg. (410, 470)
    session_name : str, optional
        Session or animal name. The default is ''. i.e: 'A1'
    axs : TYPE, optional
        The default is None.

    Returns
    -------
    A figure with the raw data trace of the selected animal

    """
    # allows us to run this fuction for different conditions ax. is a methods of axis
    if axs is None:
        fig, axs = plt.subplots(len(ch_num), 1, sharex='all', sharey='all')
    else:
        fig = axs[0].figure
    for ix, ch_ix in enumerate(ch_num):
        for w in ch_wvl:
            name = c.wvl_names[w]
            color = c.wvl_colors[w]
            axs[ix].plot(data['TimeStamp'], data[f'CH{ch_ix}{w}'], label=name, c=color) #FIXME: this does not work with one channel
        axs[ix].legend(loc='upper right')
        axs[ix].set_xlabel(TIME_LABEL)
        axs[ix].set_ylabel('Fluorescence')
        axs[ix].set_title(f'Data from CH{ch_ix} {session_name}')
        clean_ax(axs[ix])
    fig.set_tight_layout(True) 
    return session_name, fig


# TODO : There might be a way to generalize this function
def raw_signal_check(data, time_key='raw_time', data_key='raw', save=True):
    raw = data[data_key]
    time = data[time_key]
    n_sites = len(raw.keys())
    n_wvl = len(list(raw.values())[0].keys())
    fig, axs = plt.subplots(n_wvl, n_sites, sharex='all', figsize=(10.5, 10),
                            squeeze=False)
    for ix_col, c_site in enumerate(raw.keys()):
        c_all_wvl = raw[c_site]
        axs[0, ix_col].set_title(c_site)
        for ix_row, c_wvl in enumerate(c_all_wvl.keys()):
            c_sig = c_all_wvl[c_wvl]
            c_ax = axs[ix_row, ix_col]
            c_ax.plot(time, c_sig)
            c_ax.set_xlabel('Time (s)')
            c_ax.set_ylabel(f'F of {c_wvl}')
    fig.set_tight_layout(True)
    folder_name = data['filepath'].parent.parent
    fig.suptitle(f"{folder_name.stem}")
    if save:
        fig_path = folder_name / f'signal_check_{data_key}.png'
        fig.savefig(fig_path)
    return axs


def snippets_signal_check(t_snips, snippets, datapath, event_colors=None, save=True):
    all_events = set()
    for name, site in snippets.items():
        for sensor, snips in site.items():
            all_events.update(list(snips.keys()))
    n_events = len(all_events)
    colors = plt.cm.tab20(np.arange(n_events))
    if event_colors is None:
        event_colors = dict(zip(all_events, colors))

    n_sites = len(snippets.keys())
    n_sensors = len(list(snippets.values())[0].keys())
    fig, axs = plt.subplots(n_sensors, n_sites, sharex='all', figsize=(10.5, 10),
                            squeeze=False)
    for ix_col, c_site in enumerate(snippets.keys()):
        c_all_sensors = snippets[c_site]
        for ix_row, c_wvl in enumerate(c_all_sensors.keys()):
            c_snippets = c_all_sensors[c_wvl]
            c_ax = axs[ix_row, ix_col]
            for event_name, event_data in c_snippets.items():
                c_color = event_colors.get(event_name, 'black')
                plot_snippets(t_snips, event_data, c_ax, color=c_color, zscore=True,
                              show_ind=False, label=event_name)
            c_ax.legend()
            c_ax.set_title(f'{c_site} - {c_wvl}')
    fig.set_tight_layout(True)
    folder_name = datapath.parent.parent
    fig.suptitle(f"{folder_name.stem}")
    if save:
        fig_path = folder_name / 'signal_check_events.png'
        fig.savefig(fig_path)


def plot_snippets_old(t_snips, snippets, ax=None, color='black', zscore=False, show_ind=True, show_sem=True,
                  label=''):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_tight_layout(True)
    if zscore:
        snippets = zscore_snippets(t_snips, snippets)
    avg = snippets.mean(0)
    sem = snippets.std(0) / np.sqrt(snippets.shape[0])
    if show_ind:
        ax.plot(t_snips, snippets.T, c=color, lw=.5)
    if show_sem:
        ax.fill_between(t_snips, avg-sem, avg+sem, fc=color, alpha=.5)
    if len(t_snips) == 0:
        return
    ax.plot(t_snips, avg, c=color, lw=2, label=label)
    ax.set_xlabel('Time from event (s)')
    if not zscore:
        ax.set_ylabel(DELTA_F_F)
    else:
        ax.set_ylabel("Z-score fluorescence")
    ax.axvline(0, lw=.5, ls='--')


def plot_overlap_peaks(t_peaks, peaks, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)    
    ax.plot(t_peaks,peaks.T,color='blue',lw=1)
    ax.set_xlabel('Time from peak (s)')
    ax.set_ylabel(DELTA_F_F)
    #ax.axvline(0, lw=.5, ls='--')


def plot_overlap_peaks_comp(t_peaks, peaks, pk, events, avg=True, ax=None,):
    
    first_ev = events[0]
    
    baseline_mask = pk[:] < first_ev
    baseline_peaks = peaks[:][baseline_mask] #gathering peaks found before 1st stim
    #t_baseline = t_peaks[:][baseline_mask]
    bsl_peaks_avg = baseline_peaks.mean(0)
    bsl_peaks_sem = baseline_peaks.std(0) / np.sqrt(baseline_peaks.shape[0])
   
    stim_mask = pk[:] >= first_ev   
    stim_peaks = peaks[:][stim_mask] #gathering peaks found after 1st stim
    stim_peaks_avg = stim_peaks.mean(0)
    stim_peaks_sem = stim_peaks.std(0) / np.sqrt(stim_peaks.shape[0])
 
    if ax is None:
        
        fig, ax = plt.subplots(1, 1)       
    
    if avg:
        
        ax.plot(t_peaks,bsl_peaks_avg.T,color='k',lw=1, label = 'baseline')
        ax.plot(t_peaks,stim_peaks_avg.T,color='blue',lw=1, label = 'stim')
        ax.set_xlabel('Time from peak (s)')
        ax.set_ylabel(DELTA_F_F)
 
        ax.fill_between(t_peaks, bsl_peaks_avg - bsl_peaks_sem, bsl_peaks_avg + bsl_peaks_sem, color='k',
                        alpha=0.1)
        ax.fill_between(t_peaks, stim_peaks_avg - stim_peaks_sem, stim_peaks_avg + stim_peaks_sem, color='blue',
                        alpha=0.1)
            
    else:
        
         ax.plot(t_peaks,baseline_peaks.T,color='k',lw=1, label = 'baseline')
         ax.plot(t_peaks,stim_peaks.T,color='blue',lw=1, label = 'stim')
         ax.set_xlabel('Time from peak (s)')
         ax.set_ylabel(DELTA_F_F)
    return ax           


def plot_group_avg(t, signals, group_name, events, ev_dur, color='k', ax=None):
    """

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    signals : np.array
        DESCRIPTION.
    group_name : TYPE
        DESCRIPTION.
    events: np.ndarray
    color : TYPE, optional
        DESCRIPTION. The default is 'k'.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    ev_dur : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for ev in events:
        ax.axvspan(int(ev), int(ev) + ev_dur, alpha=.5, facecolor='lightblue', edgecolor='none')
    n_animals = signals.shape[0]
    avg = signals.mean(0)
    sem = signals.std(0) / np.sqrt(n_animals)
    ax.fill_between(t, avg - sem, avg + sem, color=color, alpha=0.1)
    ax.plot(t, avg, c=color, label=f'{group_name} (n={n_animals})')
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(DELTA_F_F)
    # ax.set_title(f'{datapath.stem}')
    # ax.set_title(f'{datapath.parent.stem}_'+f'{datapath.stem}')
    ax.legend(loc='upper right')
    clean_ax(ax)
    # FIXME
    # plt.savefig(datapath / f'D_F_F_avg_{group_name}.png')

    return ax


def plot_all_groups(t, grouped_signal, ev_dur, ax=None):
    stim_times = get_stim_times()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for group_name, group_data in grouped_signal.items():
        # Use get so that if group is undefined / wrong we still get the plot, though in yellow
        plot_group_avg(t, group_data, group_name, stim_times, ev_dur,
                       color=group_colors.get(group_name, 'y'), ax=ax, )
    ax.set_title('Avg signal per group')
    ax.legend(loc='upper right')

def plot_snippets_stims(snips, bsl_snips, tsnip, title, color='grey', show_avg=True):
    n_events = snips.shape[0]
    fig, axs = plt.subplots(int(np.sqrt(n_events)) + 1, int(np.sqrt(n_events)),
                            sharex='all', sharey='all')
    axs = axs.reshape(-1)
    fig.suptitle(title)
    snips_bsl = bsl_snips.mean(0)
    bsl_avg = snips_bsl.mean(1)
    bsl_sem = snips_bsl.std(1) / np.sqrt(snips_bsl.shape[1])
    axs[0].fill_between(tsnip, bsl_avg - bsl_sem, bsl_avg + bsl_sem, color=color,
                        alpha=0.1)
    axs[0].plot(tsnip, bsl_avg, c=color)
    snips_avg = snips.mean(2)
    snips_sem = snips.std(2) / np.sqrt(snips.shape[2])
    for ix in range(n_events):
        c_ax = axs[ix + 1]
        c_ax.axvline(0, lw=.5, ls='--')
        if show_avg:
            c_ax.fill_between(tsnip, snips_avg[ix] - snips_sem[ix],
                              snips_avg[ix] + snips_sem[ix],
                              color=color, alpha=0.1)
            c_ax.plot(tsnip, snips_avg[ix], c=color)
        else:
            c_ax.plot(tsnip, snips[ix])
    fig.set_tight_layout(True)


def plot_df(signal, session_name='', axs=None):
    """
    Plot the df/f of our data

    Parameters
    ----------
    signal: dict
        dictionary containing the data of one mouse obtained with the function
        analyze_mouse(filepath, '') (i.e: res)
    session_name : str, optional
        Session or animal name. The default is ''. i.e: 'A1'
    axs : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    if axs is None:
        fig, axs = plt.subplots(len(signal['channels'].keys()), 1, sharex=True, sharey=True,
                                figsize=(11, 8))
        axs = np.atleast_1d(axs)
    else:
        fig = axs.figure

    fig.suptitle(f'Data from {session_name}')
    ch_ix = 0
    t = signal['time']
    # old_time = signal['old_time']
    for site, ch in signal['channels'].items():
        for name, dff in ch.items():
            axs[ch_ix].plot(t, dff, c=c.wvl_colors.get(name, 'k'), label=name) # FIXME: not working with 1 channel
        axs[ch_ix].set_xlabel(TIME_LABEL)
        axs[ch_ix].set_ylabel(DELTA_F_F)
        axs[ch_ix].set_title(f'CH{ch_ix + 1}_{site}')
        axs[ch_ix].legend(loc='upper right')
        clean_ax(axs[ch_ix])
        ch_ix += 1

    fig.set_tight_layout(True)
    return fig


def plot_all_df(signals):
    """
    Plots df of all mice contained in all_res

    Parameters
    ----------
    signals : list
        List of dictionaries containing the data of all mice,
        obtained with the function analyze_all_mice().

    Returns
    -------
    A dff plot for each animal in the dict

    """
    for mice in signals:
        mouse_name = mice['mouse']
        fig = plot_df(mice, mouse_name)
        # FIXME
        fig.savefig(mice['filepath'].parent / f'D_F_F_{mouse_name}.png')


# plot all stimulations (Peri stim data), or one group of animals
def plot_stims(t, signals, group_name, events, t_left=10, t_right=180, avg=False, axs=None):
    """

    Parameters
    ----------
    t: np.ndarray
    signals: np.ndarray
    group_name: str
    events: np.ndarray
    t_left: float
    t_right: float
    avg: boolean
        True: plots avg data instead of individual cases
        False: plots all animals
    axs

    Returns
    -------
    Plot with signal for each stimulus during the selected period of time

    Examples
    --------
    >>> plot_stims(t, APP, 'APP', events, t_left=10, t_right=20, avg= True)
    >>> plot_stims(t, WT, 'WT', events, t_left=10, t_right=20, avg= True)

    """
    n_events = len(events)
    if axs is None:
        fig, axs = plt.subplots(int(np.sqrt(n_events)) + 1, int(np.sqrt(n_events)),
                                sharex='all', sharey='all')
        axs = axs.reshape(-1)
    else:
        fig = axs[0].figure

    starts = events - t_left
    stops = events + t_right
    start_ix = np.searchsorted(t, starts)
    stop_ix = np.searchsorted(t, stops)

    for ix in range(n_events):
        print(ix)
        ax = axs[ix]
        c_start = int(start_ix[ix])
        c_stop = int(stop_ix[ix])
        sig_slice = signals[:, c_start:c_stop]
        t_slice = t[c_start:c_stop] - events[ix]
        if not avg:
            ax.plot(t_slice, sig_slice.T)
        else:
            n_animals = len(sig_slice)
            avg_slice = sig_slice.mean(0)
            sem = sig_slice.std(0) / np.sqrt(n_animals)
            ax.fill_between(t_slice, avg_slice - sem, avg_slice + sem,
                            color=group_colors.get(group_name, 'y'), alpha=0.1)
            ax.plot(t_slice, avg_slice, color=group_colors.get(group_name, 'y'))

        ax.set_xlabel('Time from stimulation (s)')
        ax.axvline(0, lw=.5, ls='--')
        ax.set_ylabel('Delta F / F')
        fig.suptitle('Responses to consecutive stimuli_' + f'{group_name}')
        fig.set_tight_layout(True)
        # plt.savefig(datapath/ 'Responses to consecutive stimuli_'f'{group_name}.png')

    return axs




def plot_comparison(populations, names=None, colors=None,
                    linked=False, ax=None, offset=.07):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    n_pop = len(populations)
    if linked and n_pop > 2:
        warnings.warn('More than 2 linked populations, it is not going to be pretty')
    if names is None:
        names = [f'Population {ix + 1}' for ix in range(n_pop)]
    if colors is None:
        colors = cmap(np.linspace(0, 1, n_pop))
    for ix, pop in enumerate(populations):
        if len(pop) == 0:
            populations[ix] = np.array([np.nan])
    n_ind = [len(pop) for pop in populations]

    eq_pop = all([n_ind[0] == n for n in n_ind[1:]])
    xs = [np.zeros(n) + ix for ix, n in enumerate(n_ind)]
    point_pos = np.array([c_x[0] for c_x in xs])
    if linked:
        positions = [-.1] + [ix + (offset * n_pop) for ix in range(1, n_pop)]
    else:
        positions = point_pos + (offset * n_pop)
    bp = ax.boxplot(populations, patch_artist=True, positions=positions, widths=.3)
    for ix in range(n_pop):
        bp['boxes'][ix].set_facecolor(colors[ix])
        bp['boxes'][ix].set_edgecolor('none')
        bp['medians'][ix].set_color('white')
        bp['fliers'][ix].set_marker('.')
        bp['fliers'][ix].set_markerfacecolor(colors[ix])
        bp['fliers'][ix].set_markeredgecolor('none')
    for c_x, c_y, c_col in zip(xs, populations, colors):
            ax.plot(c_x, c_y, 'o', c=c_col)
    if linked and eq_pop:
        xs = np.vstack(xs)
        all_pop = np.vstack(populations)
        ax.plot(xs, all_pop, c='.2', lw=.8)

    ax.set_xticks(np.mean(np.vstack((point_pos, positions)), 0), names)
    ax.set_xticklabels(names)
    fig.set_tight_layout(True)
    return ax


def plot_peth(snippets: dict, state=0, site_name=None, dff_colors=None,
              axs=None, title=None, label=None, scale_ind=False, show_ind=False):
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
    dff_colors: dict, optional
        Dictionary specifying the colors of each DF/F per site and sensor
        If left to None (default) colors will be generated automatically
    axs: Optional, array of Matplotlib axes
        Needs to have as many rows as events (len(snippets)) and just one column
    title: Optional, str
        Title to give to the figure
    label: Optional, str
        Label for the plotted curve. If left to None, defaults to the recording site name
    scale_ind: Optional, bool
        Should the scale be based on individual curves or the average (default behavior)
        Defaults to False
    show_ind: Optional, bool
        Should the individual events be plotted or just the average?
        Defaults to False

    """
    if len(snippets) == 0:
        return None, None
    events_names = list(snippets.keys())
    if site_name is None:
        sites_names = list(snippets[events_names[0]][state].keys())
    else:
        sites_names = [site_name]
    sensors_names = list(snippets[events_names[0]][state][sites_names[0]].keys())
    n_sites = len(sites_names)
    n_sensors = len(sensors_names)
    n_events = len(snippets)
    if n_events == 0 or n_sensors == 0:
        return None, None
    if axs is None:
        fig, axs = plt.subplots(n_events, n_sites * n_sensors, sharex='all', sharey='all',
                                squeeze=False)
        fig.set_tight_layout(True)
    else:
        fig = axs.reshape(-1)[0].figure
    if dff_colors is None:
        dff_colors = generate_dff_colors(sites_names, sensors_names)

    for ix_event, c_event in enumerate(events_names):
        for ix_site, c_site in enumerate(sites_names):
            for ix_sensor, c_sensor in enumerate(sensors_names):
                    snips_data = snippets[c_event][state][c_site][c_sensor]
                    c_ax = axs[ix_event, n_sensors*ix_site+ix_sensor]
                    c_ax.set_xlabel('Time from event (s)')
                    c_ax.set_ylabel(DELTA_F_F)
                    ymin, ymax = c_ax.get_ylim()
                    col = dff_colors[c_site][c_sensor]
                    if show_ind:
                        c_ax.plot(snips_data['t'], snips_data['snips'].T, c=col, lw=.3, alpha=.3) # Plot all individual traces
                    # Plot average trace
                    avg = snips_data['snips'].mean(0)
                    lbl = label if label is not None else f'{c_site}\n{c_sensor}'
                    c_ax.plot(snips_data['t'], avg, c=col, lw=2, label=lbl)

                    if scale_ind:
                        ymin, ymax = min(ymin, np.min(snips_data['snips'])), max(ymax, np.max(snips_data['snips']))
                    else:
                        ymin, ymax = min(ymin, min(avg)), max(ymax, max(avg))
                    c_ax.set_ylim((ymin, ymax))
                    # Dashed vertical line at t = 0
                    c_ax.axvline(0, linestyle='--', color='black', linewidth=0.5)
                    c_ax.set_title(f'{c_event} - {EVENT_STATES_NAMES[state]}')
                    c_ax.legend(facecolor='w', frameon=True, loc='upper right')
                    c_ax.grid(False)
    if title is not None:
        text = fig.suptitle(title, fontsize=14)
        text.set_wrap(True)
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) #rect = [left, bottom, right, top]
    return fig, axs


def plot_events(events, t_max=None, c=(.3, .3, .3, .5), event_name='', ax=None):
    """
    Plot events as vertical bars
    Useful to examine the behavior in relation with the photometry recording

    Parameters
    ----------
    events: dict or list
        If dict, keys must be 0 and 1.
        0 contains event starts, 1 events stops. This matches the data format from the photometry files
        as well as the way we read them
    t_max :
    c: color spec
    event_name: str
        Name of the event, for the legend
    ax:  Optional, Matplotlib axes

    Returns
    -------
    ax:  Matplotlib axes

    """
    if len(events) == 0:
        return ax
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.set_xlabel('Time (s)')

    labeled = False
    if 1 not in events:
        events[1] = np.array(events[0].shape) + t_max
    for start, stop in zip(events[0], events[1]):
        if not labeled and event_name != '':
            ax.axvspan(start, stop, ymin=.1, ymax=.9, fc=c, ec='none', label=event_name)
            labeled = True
        else:
            ax.axvspan(start, stop, ymin=.1, ymax=.9, fc=c, ec='none')

    return ax


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
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(DELTA_F_F)
    return ax


def plot_signals(rec, dff_colors=None, events_colors=None, title=None, axs=None):
    """
    Figure summarizing the recording. Shows the signal as well as the events

    Parameters
    ----------
    rec: dict
        As returned from load_recording
    dff_colors: dict, optional
        Dictionary specifying the colors of each DF/F per site and sensor
        If left to None (default) colors will be generated automatically
    events_colors: dict, optional
        Dictionary specifyinfg the events colors. If left to None, the colors will be picked
        automatically
    title: str, Optional
        Title of the figure.
        If left to None, will be generated from the recoding metadata
    axs: Optional, Matplotlib axes
        As many rows as recording sites, 1 column

    """
    site_names = list(rec['dff'].keys())
    n_sites = len(site_names)
    sensor_names = list(rec['dff'][site_names[0]])
    n_sensors = len(sensor_names)
    n_subplots = int(n_sites * n_sensors)
    if title is None:
        title = merge_infos(rec, field_sep=' ', value_sep=': ')
    if dff_colors is None:
        dff_colors = generate_dff_colors(site_names, sensor_names)

    if events_colors is None:
        colors = ev_cmap(range(len(rec['events'].keys())))
        events_colors = {ev_name: colors[ix] for ix, ev_name in enumerate(rec['events'].keys())}
    if axs is None:
        fig, axs = plt.subplots(n_subplots, 1, sharex='all', sharey='all', squeeze=False,
                                figsize=(13, 5))
        axs = np.atleast_1d(np.squeeze(axs))
        fig.set_tight_layout(True)
    else:
        fig = axs.reshape(-1)[0].figure
    fig.suptitle(title)
    ax_ix = 0
    for c_site in site_names:
        for c_sensor in sensor_names:
            c_ax = axs[ax_ix]
            ax_ix += 1
            data = rec['dff'][c_site][c_sensor]

            plot_signal(rec['time'], data, label=f'{c_site}: {c_sensor}',
                        c=dff_colors[c_site][c_sensor], ax=c_ax)
            for ev_name, ev_ts in rec['events'].items():
                plot_events(ev_ts, rec['time'][-1], c=events_colors.get(ev_name, 'k'),
                            event_name=ev_name, ax=c_ax)
            c_ax.legend(facecolor='w', frameon=True, loc='upper right')
    return fig, axs


def generate_dff_colors(site_names, sensor_names):
    n_colors = len(sensor_names) * len(site_names)
    colors = cmap(range(n_colors))
    dff_colors = {c_site: {c_sensor: colors[ix_sensor * len(site_names) + ix_site]
                           for ix_sensor, c_sensor in enumerate(sensor_names)}
                  for ix_site, c_site in enumerate(site_names)}
    return dff_colors


def plot_recording(rec, snippets, state=0, dff_colors=None,
                   events_colors=None, title=None, show_ind=False,
                   save_dir=None):
    """
    Makes the full plot with the recording and its events,
    or all recording sites as well as all PETH for all events, and all recording sites.

    Parameters
    ----------
    rec: dict
        As returned from photometry.analysis.load_recording
    snippets: dict
        As returned from photometry.analysis.extract_all_snippets
    state: int
        0 or 1. Depends on if we look at the start (0) or stop (1) of the event
    dff_colors: dict, optional
        Dictionary specifying the colors of each DF/F per site and sensor
        If left to None (default) colors will be generated automatically
    events_colors: dict, optional
        Dictionary specifyinfg the events colors. If left to None, the colors will be picked
        automatically
    title: str, Optional
        Title of the figure.
        If left to None, will be generated from the recoding metadata
    show_ind: Optional, bool
        Should the individual events be plotted or just the average?
        Defaults to True
    save_dir: Optional, str or Path
        Path to save the figures. If None, then don't save

    Returns
    -------
    sig_fig: Matplotlib figure
    peth_fig: Matplotlib figure

    """
    if title is None:
        title = merge_infos(rec, field_sep=' ', value_sep=': ')
         # Plotting raw signals
    n_sites = len(rec['dff'])
    if n_sites == 0:
        return
    sig_fig, _ = plot_signals(rec, dff_colors=dff_colors, events_colors=events_colors,
                                    title=title)
    # Plotting PETHs

    peth_fig, _= plot_peth(snippets, state=state, dff_colors=dff_colors,title=title,
                           show_ind=show_ind)

    if save_dir is not None:
        save_dir = Path(save_dir)
        fig_name = merge_infos(rec)
        sig_fig_name = f"{fig_name}_DFF.png"
        peth_fig_name = f"{fig_name}_PETH.png"
        sig_fig.savefig(save_dir / sig_fig_name)
        if peth_fig is not None:
            peth_fig.savefig(save_dir / peth_fig_name)
    return sig_fig, peth_fig


def add_columns_list_df(c_data, c_filter):
    full_c_data = []
    for c_data_bp in c_data:
        for k, v in c_filter.items():
            c_data_bp[k] = v
        full_c_data.append(c_data_bp)
    return full_c_data




