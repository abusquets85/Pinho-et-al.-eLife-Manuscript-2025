# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:14:05 2021

@author: Júlia Pinnho
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd
from utilities.settings import upaths
from tqdm import tqdm
import pickle as pck
from pims import Video
try:
    import cv2
    CV_DEFINED = True
except ModuleNotFoundError:
    CV_DEFINED = False
# Note: Camera induces a lot of warpping


def find_video(dlc_filepath):
    folder = dlc_filepath.parent
    all_avi_files = folder.glob('*.avi')
    for movie in all_avi_files:
        movie_name = movie.stem
        if movie_name in dlc_filepath.stem:
            return movie


def find_dat_file(dlc_filename, poly_folder):
    all_dat_files = poly_folder.rglob('*.dat')
    for dat_file in all_dat_files:
        dat_name = dat_file.stem
        if dat_name in dlc_filename:
            return dat_file


def get_events(dat_path):
    """
    Read Imetronic dat file

    Parameters
    ----------
    dat_path: str or Path
        Path to the dat file

    Returns
    -------
    events: np.ndarray
        Events from Imetronic file
    """
    with open(dat_path, 'r') as fp:
        n_lines_header = 0
        for ix, line in enumerate(fp):
            n_lines_header += 1
            line = line.strip()
            # Remi: I would like to find something better than this extra condition
            # This is just making sure we don't fall for the missing path issue
            if line == '' and ix > 2:
                break
    events = np.genfromtxt(dat_path, delimiter='\t', skip_header=n_lines_header, dtype=int)
    return events


def read_rec_duration(events, event_num=11):
    eleven = events[events[:, 1] == event_num]
    last_ts = eleven[-2:, 0]
    is_ttl = np.any(events[:, 1] == 15)  # TTL event should be type 15
    return last_ts, is_ttl


def open_data(dlc_path, poly_folder=None, bodypart='center', likelihood_th=0.98):
    """
    Open data from DLC, polyboxes as well as the corresponding video
     
    Parameters
    ----------
    dlc_path: Path or str
        Path to the DLC file to open
    poly_folder: Optional, Path or str
        Path to the main folder containing dat files from polyboxes
    bodypart: str
        Which bodypart to load from the DLC file
    likelihood_th: float
        Threshold value on the likelihood given by DLC

    Returns
    -------
    last_ts: np.ndarray
        2 elements, int, with the total duration of polybox recording.
        Not sure about which one to use, but using the first one for now
    frame_rate: int
        Frame rate of the video (vary between 20 and 30)
    missing_frames: int
        How many frames are missing from the beginning (we assume) of the video
        Those frames are also missing rows from the DLC data
    x: np.ndarray
        X coordinates of the bodypart. Starts with a buch of nans, as many as missing frames
    y: np.ndarray
        Y coordinates of the bodypart. Starts with a buch of nans, as many as missing frames
    """
    dlc_path = Path(dlc_path)
    filename = dlc_path.stem

    video_path = find_video(dlc_path)
    v = Video(video_path)
    frame_rate = int(v.frame_rate)
    n_frames = len(v)
    v.close()
    dlc_data = open_dlc_data(dlc_path, bodypart, likelihood_th)
    assert n_frames == dlc_data[0].shape[0]
    # Imetronic data loading and syncing
    if poly_folder is not None:
        poly_folder = Path(poly_folder)
        dat_path = find_dat_file(filename, poly_folder)
        events = get_events(dat_path)
        last_ts, _ = read_rec_duration(events)
        missing_frames = int((last_ts[0] / 1000) * frame_rate - n_frames)
        if missing_frames < 0:
            print(f'Extra Imetronic data ({missing_frames}) compared to the video')
        missing_frames = np.clip(missing_frames, 0, np.inf).astype(int)
        nans = np.zeros(missing_frames) + np.nan
        dlc_data = tuple([np.hstack((nans, c)) for c in dlc_data])
    else:
        last_ts = -1
        missing_frames = 0
    return last_ts, frame_rate, missing_frames, dlc_data[0], dlc_data[1]


def open_dlc_data(data_path, bodypart='center', likelihood_th=0.98):
    """
    Open a file
    how to run: 
        open_data(data_path, bodypart='center')
        
    Parameters:
    ----------
    data_path
    path of the folder where the output data of Deeplabcut are
    
        

    Returns
    -------
    It will open the file to be analysed

    """
    # Done: Given the option to chose a body part and return only the data related to it
    with open(data_path, 'r') as fp:
        _ = fp.readline()
        parts = fp.readline().split(',')
    parts = [p.strip() for p in parts]
    if bodypart not in parts:
        raise ValueError(f'{bodypart} is not a valid bodypart')
    cols = [ix for ix, p in enumerate(parts) if p == bodypart]
    # Open a file in data_path
    dlc_data = np.genfromtxt(data_path, delimiter=',', skip_header=3)
    dlc_data = dlc_data[:, cols]
    coords = likelihoodtreshold(dlc_data, False, likelihood_th)

    return coords


def nan_removal(sig: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Perform a spline interpolation on a signal to remove missing values

    Parameters
    ----------
    sig: np.ndarray
        Signal with missing values
    t: np.ndarray
        Time vector

    Returns
    -------
    sig: np.ndarray
        Signal with interpolated values and no more nans
    """
    sig = sig.copy()
    is_nan = np.isnan(sig)
    if not np.any(is_nan):
        return sig
    is_not_nan = np.logical_not(is_nan)
    sig_correct = sig[is_not_nan]
    t_correct = t[is_not_nan]
    t_nan = t[is_nan]
    spline = splrep(t_correct, sig_correct, s=0)
    filling = splev(t_nan, spline)
    sig[is_nan] = filling
    return sig


def likelihoodtreshold(dlc_data, show=False, likelihood_th=0.98):
    """
    treshold data using likelihood
    how to run:
        likelihoodtreshold(dlc_data, show=False, likelihood_th=0.98)
    
    Parameters:
    ----------
    dlc_data
    The file to be analised
    

    Returns
    -------
   The x and y coordenates treshold using the likelood lower than 0.5

    """
    # Done: Diagnostic plot
    # treshold according likelihood the outliers track points in xx coordinate
    # Done: Simplify using masking with boolean arrays
    data = dlc_data.copy()
    # data below the likehood treshold replaced by nan
    data[dlc_data[:, 2] < likelihood_th, :2] = np.nan
    nan_frames = np.any(np.isnan(data[:, :2]), axis=1)
    t = np.arange(data.shape[0])
    data[:, 0] = nan_removal(data[:, 0], t)  # xx data being interpolated by nan removal function
    data[:, 1] = nan_removal(data[:, 1], t)  # xx data being interpolated by nan removal function
    if show:
        plt.plot(data[:, 0], data[:, 1])
        if np.sum(nan_frames) > 0:
            plt.plot(data[nan_frames, 0], data[nan_frames, 1], 'rx')
    return data[:, 0], data[:, 1]


def convert_cm(xcoordinate, ycoordinate, xscale=20 / 600, yscale=15 / 450):
    """
    convert pixels in cm
    how to run:
           convert_cm(xcoordinate, ycoordinate, xscale=20 / 600, yscale=15 / 450)
    
    Parameters:
    ----------
   xcoordinate
       coordinate in x axis in pixels
   ycoordinate
       coordinate in y axis in pixels
       

    Returns
    -------
   The x and y coordenates in centimeters

    """
    x_cm = xcoordinate * xscale
    y_cm = ycoordinate * yscale
    return x_cm, y_cm


def euclidean_distance(x_cm, y_cm):
    """
    calculate eucidean distance
    how to run:
        euclidean_distance(x_cm, y_cm)

    Parameters:
    ----------
   x_cm
       coordinate in x axis in centimeters
   y_cm
       coordinate in y axis in centimeters
    
    
    Returns
    -------
   e_distance
       The euclidean distance computed using square root of x elevated to 2 +  y elevated to 2 ,
       where x an y are the difference between pairs of frames in x_cm and y_cm,

    """
    # calculate the euclidean distance between each pairs of frames
    x = np.diff(x_cm, axis=0)
    y = np.diff(y_cm, axis=0)
    e_distance = np.sqrt((x * x) + (y * y))
    return e_distance


def speed(e_distance, framerate=25):
    """
    calculate speed
    how to run:
        speed(e_distance, framerate=25)
    
    Parameters
    ----------
    e_distance
        the euclidean distance
    framerate
    

    Returns
    -------
    velocity: np.ndarray
        the velocity for frame in mm/s

    -------

    """
    convertcm_mm = 10
    # Done: Parameter
    # calculate mean velocity
    velocity = e_distance / (1 / framerate) * convertcm_mm
    # Done: Maybe this should be velocity = Edistance / (1 / framerate)
    return velocity


def is_freezing(e_distance, dist_th=0.02):
    """
    Freezing detection based on thresholding the e_distance travelled during a pair of frames
    how to run:
        is_freezing(e_distance, dist_th=0.02)

    Parameters
    ----------
    e_distance: np.array
    dist_th: float
        Distance threshold in cm

    Returns
    -------
    freezing: boolean np.ndarray
        Indicates if the animal is freezing

    """
    return e_distance < dist_th


def binning(measure, bin_duration=60, win_duration=20, framerate=25, agg_func=np.sum):
    """
    Bin a variable with a bin duration of 60 seconds, for a window duration of 20 minutes,
    framerate of 20 and function default sum
    how to run:
        binning(velocity, bin_duration=60, win_duration=20, framerate=25, agg_func=np.sum)


    Parameters
    ----------
    measure: it could be any variable to be analysed by bins - np.ndarray
    bin_duration :  bin duration of 60 seconds
    win_duration : duration of the experiment (20 minutes)
    framerate : frame rate of the videos (25 fps)
    agg_func : default function

    Returns
    -------
    arr_meas: bins np.array

    """
    bin_size = bin_duration * framerate
    frame_per_min = 60 * framerate
    bins = np.arange(0, win_duration * frame_per_min, bin_size)
    l_meas = []
    for c_bin in bins:
        binned_meas = measure[c_bin:c_bin + bin_size]
        if len(binned_meas) > 0:
            binned_measure = agg_func(binned_meas)
        else:
            binned_measure = np.nan
        l_meas.append(binned_measure)
    arr_meas = np.array(l_meas) 
    return arr_meas


def freezing_periods(is_freezed, min_duration=2, framerate=25):
    w, = np.nonzero(np.diff(is_freezed))
    n_tr = len(w)
    if n_tr == 0:
        return np.array([])
    if n_tr % 2:
        if is_freezed[-1]:
            w = np.hstack((w, len(is_freezed)))
        else:
            w = np.hstack((0, w))
    periods = w.reshape((-1, 2))
    durations = np.squeeze(np.diff(periods, axis=1))
    gi = durations > (min_duration * framerate)
    periods = periods[gi]
    return periods


def freezing_speed_quantif(e_distance, velocity, dist_th=0.02, framerate=25,
                           bin_duration=60, win_duration=20, min_duration=2, show=False):
    """
    Freezing and speed quantification
    How to run:
        freezing_speed_quantif(e_distance, velocity, dist_th=0.02, framerate=25,
                           bin_duration=60, win_duration=20, show=False)

    Parameters
    ----------
    e_distance : instantant eucleadian distance - np. array
    velocity : instant velocity - np.array
    dist_th : freezing threshold (0.02)
    framerate : frame rate of the videos (25 fps)
    bin_duration : bin duration of 60 seconds
    win_duration : duration of the experiment (20 minutes)
    min_duration: float
        Minimum duration of immobility, in seconds, before it is considered freezing
    show: boolean
        Plot or not? Default to False

    Returns
    -------
    res: dict
        dur_freezing: Total duration of freezing in seconds (float)
        freezing_bin : freezing by bins - np.array
        speed: Velocity - np.ndarray
        speed_bin : speed by bins - np.array
        speed_freezing : speed during freezing
        speed_moving : speed out of the freezing period
        perc_freezing: Percentage of total time spent freezing - float

    """
    n_frames = len(e_distance)
    freezing = is_freezing(e_distance, dist_th)
    empty_bins = binning(np.ones(n_frames), bin_duration, win_duration, framerate, np.sum) == 0
    n_freezing_frames = binning(freezing, bin_duration, win_duration, framerate, np.sum)
    sec_freezing = n_freezing_frames / framerate  # Freezing duration, binned
    sec_freezing[empty_bins] = np.nan
    speed_bin = binning(velocity, bin_duration, win_duration, framerate, np.mean)  # Speed binned
    speed_bin[empty_bins] = np.nan
    speed_freezing = np.mean(velocity[freezing])
    speed_moving = np.mean(velocity[np.logical_not(freezing)])
    perc_freezing = 100 * np.nansum(n_freezing_frames) / n_frames
    no_freezing_velocity = velocity.copy()
    no_freezing_velocity[freezing] = np.nan
    speed_moving_bin = binning(no_freezing_velocity, bin_duration, win_duration, framerate, np.nanmean) #speed moving binned
    speed_moving_bin[empty_bins] = np.nan  
    if show:
        speed_color = 'darkred'
        freezing_color = 'darkblue'
        fig, ax_fr = plt.subplots()
        t = np.arange(0, win_duration * 60, bin_duration)
        ax_sp = ax_fr.twinx()
        ax_fr.plot(t, sec_freezing, c=freezing_color)
        ax_fr.set_ylabel('Binned time (s)', color=freezing_color)
        ax_fr.tick_params(axis='y', labelcolor=freezing_color)
        ax_sp.plot(t, speed_bin, c=speed_color)
        ax_sp.set_ylabel('Animal velocity (mm/s)', color=speed_color)
        ax_sp.tick_params(axis='y', labelcolor=speed_color)
        fig.set_tight_layout(True)
    res = {'dur_freezing': np.nansum(sec_freezing), 'freezing_bin': sec_freezing*100/bin_duration,
           'velocity': velocity, 'speed_bin': speed_bin, 'speed_freezing': speed_freezing,
           'speed_moving': speed_moving, 'perc_freezing': perc_freezing, 'speed': np.mean(velocity),
           'distance': np.sum(e_distance), 'freezing': freezing, 'dist': e_distance,
           'freeze_periods': freezing_periods(freezing, min_duration=min_duration, framerate=framerate),
           'speedmoving_bin': speed_moving_bin, } # attention dont call binned variables like speed_moving_bin
    # being that the function of analyse all data is searching for _bin wil skip this variable because have a _ before_bin
    return res


def split_by_freezing(measurement, freezing):
    """
    this function split a variable during freezing and moving periods

    Parameters
    ----------
    measurement: np.ndarray
        variable to be splitted
    freezing: np.ndarray
        Boolean variable that contain the information if the animal is in freezing for each frame

    Returns
    --------
    averages: dict
        the average of the variable (in measurement) during freezing and moving
        {'avg': np.ndarray, 'freezing': np.ndarray, 'moving': np.ndarray}

    -------

    """
    if len(measurement) == len(freezing) + 1:
        measurement = measurement[1:]
    in_freezing = measurement[freezing]  # filter the variable to the moments in freezing
    out_freezing = measurement[
        np.logical_not(freezing)]  # filter the variable to the moments moving
    averages = {'avg': np.nanmean(measurement),  # average of the variable without filters
                'freezing': np.nanmean(in_freezing),  # average of the variable filtered by freezing
                'moving': np.nanmean(out_freezing)}  # average of the variable filtered by moving
    return averages


def pose(data_path, poly_folder, dist_th=0.02):
    """
    Calculate the pose of the animal (each bodyparts distance or all body) during all time
    or splited by freezing or moving time
    how to run:
        create_velocity_map(results, bin_size=.5)

    Parameters
    ----------
    data_path: variable to be splitted
    dist_th : freezing treshold (0.02)

    Returns
    --------
    pose_results: dict.

    """
    data_path = Path(data_path)
    bodyparts = ('nose', 'head', 'center', 'tail')  # bodyparts to be extracted
    part_pairs = list(zip(bodyparts, bodyparts[1:]))  # iterate 2 list to do the pairs of bodyparts
    bp_pos = {}
    # for each bodypart open the file, interpolate by likelihood and convert coordinate pixels in cm
    for bp in bodyparts:
        _, _, _, x, y = open_data(data_path, poly_folder, bp, likelihood_th=0.99)
        x, y = convert_cm(x, y)
        bp_pos[bp] = (x, y)

    bp_dist = {}
    total_pose = np.zeros(
        len(bp_pos['nose'][0]) - 1)  # create the total pose with the proper size filled of zeros
    for pair in part_pairs:  # calculate the euclidean distance between each pair of body parts
        p1 = bp_pos[pair[0]]
        p2 = bp_pos[pair[1]]
        dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        bp_dist[f'{pair[0]}-{pair[1]}'] = dist[1:]  # Removing first frame, to match freezing length
        total_pose = total_pose + dist[1:]  # compute the total distance adding all the pairs
    e_distance = euclidean_distance(bp_pos['center'][0], bp_pos['center'][1])
    freezing = is_freezing(e_distance, dist_th)
    total_pose_avg = split_by_freezing(total_pose, freezing)
    bp_len_freezing = {}
    # compute the average for each bodypart length splitting by freezing and moving
    for pair, length in bp_dist.items():
        avgs = split_by_freezing(length, freezing)
        bp_len_freezing[pair] = avgs
    bp_len_freezing['total'] = total_pose_avg
    pose_results = {'bodyparts_lengths': bp_dist, 'animal_length': total_pose,
                    'pose_freezing': bp_len_freezing}
    return pose_results


def save_data(results, data_file):
    """

    Parameters
    ----------
    results: dict
        Dictionary with computed data as returned from analyze_mouse
    data_file: Path

    Returns
    -------
    outputfile with all this data
    format .npz

    -------

    """
    # Done: Change name according to raw data file name
    # create a output file with the data

    output_file = get_saved_filepath(data_file)
    with open(output_file, 'wb') as pf:
        pck.dump(results, pf)


def get_saved_filepath(data_path):
    output_file = data_path.parent / f'{data_path.stem}.pck'
    return output_file


def validate_saved(data_path, force, parameters):
    output_file = get_saved_filepath(data_path)
    saved_ok = False
    res = {}
    if not force and output_file.exists():
        with open(output_file, 'rb') as pf:
            res = pck.load(pf)
        saved_ok = True
        if 'params' in res.keys():
            for prm_name, value in parameters.items():
                if res['params'][prm_name] != value:
                    saved_ok = False
                    break
        else:
            saved_ok = False
    return saved_ok, res


def analyze_mouse(data_path, poly_folder=None, bodypart='center', likelihood_th=0.98,
                  dist_th=0.02, bin_duration=60, win_duration=20, min_duration=2, force=False):
    data_path = Path(data_path)
    parameters = {'dist_th': dist_th, 'bin_duration': bin_duration,
                  'win_duration': win_duration, 'min_duration': min_duration}
    saved_ok, res = validate_saved(data_path, force, parameters)
    if saved_ok:
        return res
    if not data_path.exists():
        return {}
    _, framerate, _, x, y = open_data(data_path, poly_folder, bodypart, likelihood_th)
    dt = 1 / framerate
    time = np.arange(0, x.shape[0] * dt, dt)
    x, y = convert_cm(x, y)
    dist = euclidean_distance(x, y)
    velocity = speed(dist, framerate)
    res = freezing_speed_quantif(dist, velocity, dist_th, framerate, bin_duration, win_duration,
                                 min_duration)
    # Add coordinates into result dictionary
    res['x'] = x
    res['y'] = y
    # Also, time
    res['time'] = time
    #add a slice of freezing # meas variable in percentage # time variable duration of the slice
    slice = slice_measurement(res['time'], res['freezing'], [180, ])
    res['time_slices1'] = len(slice[0][0])/framerate
    res['time_slices2'] = len(slice[0][1])/framerate
    res['meas_slices1'] = (((slice[1][0].sum())/framerate)/res['time_slices1'])*100
    res['meas_slices2'] = (((slice[1][1].sum())/framerate)/res['time_slices2'])*100
    #res['time_in_ROI'] = in_ROi(data_path, mice_path, base_path=upaths['datapath'], bp='nose', framerate=25)
    # Pose analysis
    pose_res = pose(data_path, poly_folder, dist_th)
    res['pose'] = pose_res
    fused_pose = merge_pose_freezing(pose_res['pose_freezing'])
    for k, v in fused_pose.items():
        res[k] = v
    #v_map = create_velocity_map(res)
    #res['v_map'] = v_map

    res['params'] = parameters
    save_data(res, data_path)
    return res


def merge_pose_freezing(pose_freezing):
    """
    dictionary fusing the pose data, reduction of hierarchy
    (Pose-> head -> avg   will be avg_head_pose)
    Parameters
    ----------
    pose_freezing:
        avg length to each bodypart during freezing and moving

    Returns
    -------
    fused: dict
        the same variables in a dictionary with a reduction of the hierarchy
    """
    fused = {}
    for bodyparts, avgs in pose_freezing.items():
        for key, value in avgs.items():
            fused[f'{bodyparts}_{key}'] = value
    return fused


def analyse_all_data(mice_path, poly_folder, base_path=upaths['dlcpath'], force=False):
    """
    Search in a directory csv files and run all the functions inside the for cycle in those files
    export a output_file with FreezingTime,FreezingBinsMin,Velocity,output_file

    Parameters
    ----------
    mice_path: str
        Path to a CSV (eventually XLS later) containing info on experiments and files
    poly_folder: str or Path
        Path to the directory containing all polyboxes files
    base_path: str or Path
        Path to the directory where data are to be found
    force: bool
        Recompute or load from disk?

    Returns
    -------
    df: pandas.DataFrame
        Contains input data from the excel file + computed parameters
    -------

    """
    base_path = Path(base_path)
    df = pd.read_csv(mice_path)
    for ix_row, row in tqdm(df.iterrows()):
        data_path = base_path / (row['file'] + '.csv')
        r = analyze_mouse(data_path, poly_folder, force=force)
        fill_table(df, ix_row, r)
    mice_path = Path(mice_path)
    df.to_excel(mice_path.parent / 'all_computed_data.xlsx')
    return df


def slice_measurement(time, meas, event_times):
    """
    slice a variable according with a specific time
    how to call:
       slice_measurement(results['time'], results['perc_freezing'], [60,])

    Parameters
    ----------
    time: np.ndarray
        Time vector of the entire recording
    meas: np.ndarray
        variable to be sliced
    event_times: list, tuple or np.ndarray
        time of events. We will add the begining and the end of the recording if not
        already present time in seconds where you want to slice

    Returns
    -------
    time_slices: list
        duration of the slice
    meas_slices: list
        the value of the variable sliced


    """
    event_times = np.sort(event_times)
    if event_times[0] != time[0]:
        event_times = np.hstack(([0], event_times))
    if event_times[-1] != time[-1]:
        event_times = np.hstack((event_times, time[-1]))
    ix_evt = np.searchsorted(time, event_times)
    meas_slices = []
    time_slices = []
    for start, stop in zip(ix_evt[:-1], ix_evt[1:]):
        # iterate through all the events List 1 = all excluding last list 2 = all excluding first
        m_slice = meas[start:stop]  # compute the variable sliced for each slice
        t_slice = time[start:stop]  # compute the time sliced for each slice
        meas_slices.append(m_slice)
        time_slices.append(t_slice)
    return time_slices, meas_slices


def fill_table(df, ix_row, results):
    """
    Given the results from the analysis of one file, fill them in the global table
    
    Parameters
    ----------
    df
    ix_row
    results

    Returns
    -------

    """
    for key, value in results.items():
        if isinstance(value, np.ndarray) and '_bin' in key:
            # Deal with arrays by adding a column in the DF for each column in the array
            column_base = key.split('_')[0]
            for bin_num in range(len(value)):
                col_name = f'{column_base}_{bin_num + 1}'
                if col_name not in df.columns:
                    df[col_name] = np.nan
                df.loc[ix_row, col_name] = value[bin_num]
        elif not isinstance(value, np.ndarray) and not isinstance(value, dict):
            if key not in df.columns:
                df[key] = np.nan
            df.loc[ix_row, key] = value


if __name__ == '__main__' and False:
    df = analyse_all_data(upaths['table_path'], upaths['table_path'])
    # df['group']= df['brain']+df['drug']
    
    # manually analyzing one file
#    d = open_data(upaths[
#                      'datapath'] / '20210808 _ERC project_JP_tone context_habituation ab 2_01_01_1DLC_resnet50_FearDetectionJun17shuffle1_100000.csv')
#    x, y = likelihoodtreshold(d)
#    x, y = convert_cm(x, y)
#    dist = euclidean_distance(x, y)
#    velocity = speed(dist)
#    freezing = is_freezing(dist)
#    results = freezing_speed_quantif(dist, velocity)
    # Can be replaced with:
#    results = analyze_mouse(upaths[
#                                'datapath'] / '20210808 _ERC project_JP_tone context_habituation ab 2_01_01_1DLC_resnet50_FearDetectionJun17shuffle1_100000.csv')

    # How to analyze all files
    #  df = analyse_all_data(upaths['table_path'])
