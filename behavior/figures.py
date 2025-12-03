import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
# To test with multiple drugs
# df.drug.iloc[::2] = 'drug'


# TODO: Generalize for more parameters
def perc_freezing_boxplots(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehicle', 'drug'))
    # sns.pointplot(x='protocol', y='perc_freezing', hue='drug', data=df, hue_order=('vehicle', 'drug'))
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehiclewt', 'vehicleapp', 'aefwt','aefapp'))
    sorting_order = ['habituation', 'pc1','pc2', 'pc3', 'pc4', 'pc5', 'pc6','ptt', 'ptl','ptt1w', 'ptl1w']
    sns.boxplot(x='protocol', y='perc_freezing', data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order)
    sns.stripplot(x='protocol', y='perc_freezing', data=df, ax=ax,
                  order=sorting_order, dodge=True)
    #ymin, ymax = ax.get_ylim()
    #ax.hlines(ymax, 1, 2, lw=1, colors='k')
   # s_right = stars(0.001)
   # ax.text(1.5, ymax * 1.01, s_right, fontsize=16, verticalalignment='bottom',
    #        horizontalalignment='center')
    # ax.text(1.5, ymax * 1.01, f'p={0.0015:.4f}', fontsize=16, va='bottom')
    #ax.set_ylim(ymin, ymax*1.1)
    return ax

def perc_freezing_Off_boxplots(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehicle', 'drug'))
    # sns.pointplot(x='protocol', y='perc_freezing', hue='drug', data=df, hue_order=('vehicle', 'drug'))
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehiclewt', 'vehicleapp', 'aefwt','aefapp'))
    sorting_order = ['habituation', 'pc1','pc2', 'pc3', 'pc4', 'pc5', 'pc6','ptt', 'ptl','ptt1w', 'ptl1w']
    sns.boxplot(x='protocol', y='meas_slices1', data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order)
    sns.stripplot(x='protocol', y='meas_slices1', data=df, ax=ax,
                  order=sorting_order, dodge=True)
    #ymin, ymax = ax.get_ylim()
    #ax.hlines(ymax, 1, 2, lw=1, colors='k')
   # s_right = stars(0.001)
   # ax.text(1.5, ymax * 1.01, s_right, fontsize=16, verticalalignment='bottom',
    #        horizontalalignment='center')
    # ax.text(1.5, ymax * 1.01, f'p={0.0015:.4f}', fontsize=16, va='bottom')
    #ax.set_ylim(ymin, ymax*1.1)
    return ax

def perc_freezing_ON_boxplots(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehicle', 'drug'))
    # sns.pointplot(x='protocol', y='perc_freezing', hue='drug', data=df, hue_order=('vehicle', 'drug'))
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehiclewt', 'vehicleapp', 'aefwt','aefapp'))
    sorting_order =  ['habituation', 'pc1','pc2', 'pc3', 'pc4', 'pc5', 'pc6','ptt', 'ptl','ptt1w', 'ptl1w']
    sns.boxplot(x='protocol', y='meas_slices2', data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order)
    sns.stripplot(x='protocol', y='meas_slices2', data=df, ax=ax,
                  order=sorting_order, dodge=True)
    #ymin, ymax = ax.get_ylim()
    #ax.hlines(ymax, 1, 2, lw=1, colors='k')
   # s_right = stars(0.001)
   # ax.text(1.5, ymax * 1.01, s_right, fontsize=16, verticalalignment='bottom',
    #        horizontalalignment='center')
    # ax.text(1.5, ymax * 1.01, f'p={0.0015:.4f}', fontsize=16, va='bottom')
    #ax.set_ylim(ymin, ymax*1.1)
    return ax


def perc_freezing_3_off_boxplots(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    sorting_order = ['ptt1w', 'ptl1w']
    sns.boxplot(x='protocol', y='meas_slices1', data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order)
    sns.stripplot(x='protocol', y='meas_slices1', data=df, color='.3', ax=ax,
                  order=sorting_order)
    return ax

def perc_freezing_3_on_boxplots(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    sorting_order = ['ptt', 'ptl']
    sns.boxplot(x='protocol', y='meas_slices2', data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order)
    sns.stripplot(x='protocol', y='meas_slices2', data=df, color='.3', ax=ax,
                  order=sorting_order)
    return ax


def perc_freezing_OffOn_boxplots(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot
    during PT splitted by on and off periods

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    sorting_order1 = ['ptt', '','ptl','']
    sns.boxplot(x='protocol', y='meas_slices1', data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order1)
    sns.stripplot(x='protocol', y='meas_slices1', data=df, color='.3', ax=ax,
                  order=sorting_order1)
    sorting_order2 = ['pttOff', 'ptt','ptlOff','ptl']
    sns.boxplot(x='protocol', y='meas_slices2', data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order2)
    sns.stripplot(x='protocol', y='meas_slices2', data=df, color='.3', ax=ax,
                  order=sorting_order2)
    return ax



def perc_freezing_violinplots(df, ax=None):
    """
    Plotting percentage of freezing violin plot

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax


    """
    if ax is None:
        fig, ax = plt.subplots()
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehicle', 'drug'))
    # sns.pointplot(x='protocol', y='perc_freezing', hue='drug', data=df, hue_order=('vehicle', 'drug'))

    sorting_order = ['habituation2', 'probe test tone', 'probe test light']
    sns.violinplot(x='protocol', y='perc_freezing', data=df, showfliers=False, width=.3, ax=ax,
                   order=sorting_order)
    return ax



# to fix it return pc1 name with pc mean because i replace the data
def perc_freezing_pc_mean_boxplots(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot with preconditioning session averaged

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
#    df_hab = df[df['protocol'] == 'habituation1']
    df_new = df[df['protocol'] == 'ptt']
    df_old = df[df['protocol'] == 'ptl']
    df_pc1 = df[df['protocol'] == 'pc1']
    df_pc2 = df[df['protocol'] == 'pc2']
    df_pc3 = df[df['protocol'] == 'pc3']
    df_pc4 = df[df['protocol'] == 'pc4']
    df_pc5 = df[df['protocol'] == 'pc5']
    df_pc6 = df[df['protocol'] == 'pc6']
    df_new.set_index('mouse', inplace=True)
    df_old.set_index('mouse', inplace=True)
    df_pc1.set_index('mouse', inplace=True)
    df_pc2.set_index('mouse', inplace=True)
    df_pc3.set_index('mouse', inplace=True)
    df_pc4.set_index('mouse', inplace=True)
    df_pc5.set_index('mouse', inplace=True)
    df_pc6.set_index('mouse', inplace=True)
    df_pc_mean = df_pc1.assign(pc_mean=df_pc1['perc_freezing'] + df_pc2['perc_freezing'] + df_pc3['perc_freezing'] +
                               df_pc4['perc_freezing'] + df_pc5['perc_freezing'] + df_pc6['perc_freezing'] / 6)
    df_pc = pd.concat((df_pc_mean, df_new, df_old))
    sns.boxplot(x='protocol', y='perc_freezing', data=df_pc, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='perc_freezing', data=df_pc, color='.3', ax=ax)
    return ax


def perc_freezing_pc_mean_boxplots_on(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot with preconditioning session averaged

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
#    df_hab = df[df['protocol'] == 'habituation1']
    df_new = df[df['protocol'] == 'ptt']
    df_old = df[df['protocol'] == 'ptl']
    df_pc1 = df[df['protocol'] == 'pc1']
    df_pc2 = df[df['protocol'] == 'pc2']
    df_pc3 = df[df['protocol'] == 'pc3']
    df_pc4 = df[df['protocol'] == 'pc4']
    df_pc5 = df[df['protocol'] == 'pc5']
    df_pc6 = df[df['protocol'] == 'pc6']
    df_new.set_index('mouse', inplace=True)
    df_old.set_index('mouse', inplace=True)
    df_pc1.set_index('mouse', inplace=True)
    df_pc2.set_index('mouse', inplace=True)
    df_pc3.set_index('mouse', inplace=True)
    df_pc4.set_index('mouse', inplace=True)
    df_pc5.set_index('mouse', inplace=True)
    df_pc6.set_index('mouse', inplace=True)
    df_pc_mean = df_pc1.assign(pc_mean=df_pc1['meas_slices2'] + df_pc2['meas_slices2'] + df_pc3['meas_slices2'] +
                               df_pc4['meas_slices2'] + df_pc5['meas_slices2'] + df_pc6['meas_slices2'] / 6)
    df_pc = pd.concat((df_pc_mean, df_new, df_old))
    sns.boxplot(x='protocol', y='meas_slices2', data=df_pc, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='meas_slices2', data=df_pc, color='.3', ax=ax)
    return ax

def perc_freezing_pc_mean_boxplots_off(df, ax=None):
    """
    Plotting percentage of freezing boxplot + stripplot with preconditioning session averaged

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
#    df_hab = df[df['protocol'] == 'habituation1']
    df_new = df[df['protocol'] == 'ptt']
    df_old = df[df['protocol'] == 'ptl']
    df_pc1 = df[df['protocol'] == 'pc1']
    df_pc2 = df[df['protocol'] == 'pc2']
    df_pc3 = df[df['protocol'] == 'pc3']
    df_pc4 = df[df['protocol'] == 'pc4']
    df_pc5 = df[df['protocol'] == 'pc5']
    df_pc6 = df[df['protocol'] == 'pc6']
    df_new.set_index('mouse', inplace=True)
    df_old.set_index('mouse', inplace=True)
    df_pc1.set_index('mouse', inplace=True)
    df_pc2.set_index('mouse', inplace=True)
    df_pc3.set_index('mouse', inplace=True)
    df_pc4.set_index('mouse', inplace=True)
    df_pc5.set_index('mouse', inplace=True)
    df_pc6.set_index('mouse', inplace=True)
    df_pc_mean = df_pc1.assign(pc_mean=df_pc1['meas_slices1'] + df_pc2['meas_slices1'] + df_pc3['meas_slices1'] +
                               df_pc4['meas_slices1'] + df_pc5['meas_slices1'] + df_pc6['meas_slices1'] / 6)
    df_pc = pd.concat((df_pc_mean, df_new, df_old))
    sns.boxplot(x='protocol', y='meas_slices1', data=df_pc, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='meas_slices1', data=df_pc, color='.3', ax=ax)
    return ax

def binned_freezing(df, ax=None):
    """
    Plotting percentage of freezing boxplot with bins (in this case one minute)

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax


    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
#id_vars=['mouse', 'drug', 'protocol']
    df_melted = df.melt(id_vars=['mouse', 'protocol'],
                        value_vars=[c for c in df.columns if 'freezing_' in c],
                        value_name='Freezing percentage', var_name='Time bin')
    sns.boxplot(x='Time bin', y='Freezing percentage', hue='protocol', hue_order= ('PTT', 'PTL'), data=df_melted, ax=ax)
    sns.stripplot(x='Time bin', y='Freezing percentage', hue='protocol',hue_order= ('PTT', 'PTL'), data=df_melted, ax=ax, dodge=True)
    return ax



def speed_boxplots(df, ax=None):
    """
    Plotting speed boxplot + stripplot

    Parameters
    ----------
    df: variable that it contain all the data 
    ax: Optional, matplotlib axes
        
    Returns
    -------
    ax
 

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='speed', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='speed', data=df, color='.3', ax=ax)
    return ax


def speed_boxplot_binned(df, ax=None):
    """
    Plotting speed boxplot with bins (in this case one minute)

    Parameters
    ----------
    df: variable that it contain all the data 
    ax: Optional, matplotlib axes
    
    Returns
    -------
    ax
    
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    df_melted = df.melt(id_vars=['mouse', 'protocol'],
                        value_vars=[c for c in df.columns if 'speed_' in c],
                        value_name='speed', var_name='Time bin')

    sns.boxplot(x='Time bin', y='speed', hue='protocol', data=df_melted,
                width=.3, hue_order=('habituation', 'probe test old'), ax=ax)
    return ax



def speed_moving_boxplots(df, ax=None):
    """
    Plotting speed boxplot when animal is moving

    Parameters
    ----------
    df: variable that it contain all the data 
    ax: Optional, matplotlib axes
    
    Returns
    -------
    ax
    
    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='speed_moving', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='speed_moving', data=df, color='.3', ax=ax)
    return ax

def speed_moving_binned(df, ax=None):
    """
    Plotting percentage of freezing boxplot with bins (in this case one minute)

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax


    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    df_melted = df.melt(id_vars=['mouse', 'protocol'],
                        value_vars=[c for c in df.columns if 'speedmoving_' in c],
                        value_name='speedmoving', var_name='Time bin')

    sns.boxplot(x='Time bin', y='speedmoving', hue='protocol', data=df_melted,
                width=.3, ax=ax)
    # sns.stripplot(x='Time bin', y='speedmoving', hue='protocol', data=df_melted, ax=ax, jitter=False)
    return ax


def speed_freezing_boxplots(df, ax=None):
    """
    Plotting speed boxplot when animal is freezing

    Parameters
    ----------
    df: variable that it contain all the data 
    ax: Optional, matplotlib axes
            
    Returns
    -------
    ax


    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='speed_freezing', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='speed_freezing', data=df, color='.3', ax=ax)
    return ax


def flight_boxplot(df, ax=None):
    """
    Plotting flight score boxplot 

    Parameters
    ----------
    df: variable that it contain all the data 
    ax: Optional, matplotlib axes

    Returns
    -------
    ax


    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    df_hab = df[df['protocol'] == 'hab']
    df_new = df[df['protocol'] == 'ptt']
    df_old = df[df['protocol'] == 'ptl']
    df_hab.set_index('mouse', inplace=True)
    df_new.set_index('mouse', inplace=True)
    df_old.set_index('mouse', inplace=True)
    df_new = df_new.assign(flight=df_new['speed'] / df_hab['speed'])
    df_old = df_old.assign(flight=df_old['speed'] / df_hab['speed'])
    df_flight = pd.concat((df_new, df_old))
    sns.boxplot(x='protocol', y='flight', data=df_flight, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='flight', data=df_flight, color='.3', ax=ax)
    return ax


def pose_boxplots(df, ax=None):
    """
    Plotting avg total pose= avg (distance nose-head + head-center + center-tail)

    Parameters
    ----------
    df: variable that it contain all the data
    ax: Optional, matplotlib axes

    Returns
    -------
    ax


    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='total_avg', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='total_avg', data=df, color='.3', ax=ax)
    return ax


def pose_freezing_boxplots(df, ax=None):
    """
    Plotting avg total pose = avg (distance nose-head + head-center + center-tail) during freezing

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='total_freezing', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='total_freezing', data=df, color='.3', ax=ax)
    return ax



def pose_moving_boxplots(df, ax=None):
    """
    Plotting avg total pose = avg (distance nose-head + head-center + center-tail) during moving

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='total_moving', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='total_moving', data=df, color='.3', ax=ax)
    return ax




def nose_head_boxplots(df, ax=None):
    """
    Plotting avg distance nose-head

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='nose-head_avg', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='nose-head_avg', data=df, color='.3', ax=ax)
    return ax



def nose_head_freezing_boxplots(df, ax=None):
    """
    Plotting avg distance nose-head during freezing

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='nose-head_freezing', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='nose-head_freezing', data=df, color='.3', ax=ax)
    return ax



def nose_head_moving_boxplots(df, ax=None):
    """
    Plotting avg distance nose-head during moving

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='nose-head_moving', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='nose-head_moving', data=df, color='.3', ax=ax)
    return ax



def head_center_boxplots(df, ax=None):
    """
    Plotting avg distance head_center

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='head-center_avg', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='head-center_avg', data=df, color='.3', ax=ax)
    return ax



def head_center_freezing_boxplots(df, ax=None):
    """
    Plotting avg distance head_center during freezing

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='head-center_freezing', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='head-center_freezing', data=df, color='.3', ax=ax)
    return ax



def head_center_moving_boxplots(df, ax=None):
    """
    Plotting avg distance head-center during moving

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='head-center_moving', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='head-center_moving', data=df, color='.3', ax=ax)
    return ax


def center_tail_boxplots(df, ax=None):
    """
    Plotting avg distance center_tail

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='center-tail_avg', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='center-tail_avg', data=df, color='.3', ax=ax)
    return ax



def center_tail_freezing_boxplots(df, ax=None):
    """
    Plotting avg distance center_tail during freezing

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='center-tail_freezing', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='center-tail_freezing', data=df, color='.3', ax=ax)
    return ax


def center_tail_moving_boxplots(df, ax=None):
    """
    Plotting avg distance center-tail during moving

    Parameters
    ----------
    df: variable that it contain all the data

    Returns
    -------
    ax
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x='protocol', y='center-tail_moving', data=df, showfliers=False, width=.3, ax=ax)
    sns.stripplot(x='protocol', y='center-tail_moving', data=df, color='.3', ax=ax)
    return ax



def score_freezing_speedmoving_boxplots(df, ax=None):
    """
    
    """
    if ax is None:
        fig, ax = plt.subplots()
    # sns.boxplot(x='protocol', y='perc_freezing', hue='drug', data=df, showfliers=False, width=.3, hue_order=('vehicle', 'drug'))
    # sns.pointplot(x='protocol', y='perc_freezing', hue='drug', data=df, hue_order=('vehicle', 'drug'))
    score =  df['perc_freezing'] / df['speed_moving']
    sorting_order = ['hab','ptt', 'ptl']
    sns.boxplot(x='protocol', y=score, data=df, showfliers=False, width=.3, ax=ax,
                order=sorting_order)
    sns.stripplot(x='protocol', y=score, data=df, color='.3', ax=ax,
                  order=sorting_order)
    #ymin, ymax = ax.get_ylim()
    #ax.hlines(ymax, 1, 2, lw=1, colors='k')
   # s_right = stars(0.001)
   # ax.text(1.5, ymax * 1.01, s_right, fontsize=16, verticalalignment='bottom',
    #        horizontalalignment='center')
    # ax.text(1.5, ymax * 1.01, f'p={0.0015:.4f}', fontsize=16, va='bottom')
    #ax.set_ylim(ymin, ymax*1.1)
    return ax



def correlation_graph(df, ax=None):
    x = df[df['protocol'] == 'ptt']
    y = df[df['protocol'] == 'ptl']
    plt.plot(x['perc_freezing'], y['perc_freezing'], 'o', color='black')
    return ax

def pie_charp(df, ax=None):
    y = np.array([37.5, 62.5])
    labels=['Non_learners 37.5%','Learners 62.5%']
    plt.pie(y, labels=labels)
    return ax
#to tun these individual plots run the file of the individual animal that you want to plot:
#results = analyze_mouse(upaths['datapath'] / '20210808 _ERC project_JP_tone context_habituation ab 2_01_01_1DLC_resnet50_FearDetectionJun17shuffle1_100000.csv')

def speed_map(results, ax=None):
    """
    Plotting speed map (speed plotted in x and y coordinates)

    Parameters
    ----------
    results: dict
    ax: Optional, matplotlib axes

    Returns
    -------
    ax

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x = results['x']
    y = results['y']
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    ax.imshow(results['v_map'], extent=(xmin, xmax, ymin, ymax), origin='lower')


def time_position_map(results, ax=None):
    """
    Plotting time spent map (time spent plotted in x and y coordinates)

    Parameters
    ----------
    results: dict
    ax: Optional, matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x, y = results['x'], results['y']
    counts, xedges, yedges = np.histogram2d(x, y, 80)
    ax.imshow(counts, extent=(y.min(), y.max(), x.min(), x.max()), origin='lower')
    # ax.plot(y, x, 'w', linewidth=.3, alpha=.5)


def position_speed_map(results, ax=None):
    """
    Plotting speed map and time spent map

    Parameters
    ----------
    results: dict
    ax: Optional, matplotlib axes

    Returns
    -------
    ax

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    time_position_map(results, ax)
    x, y = results['x'], results['y']
    xs = np.column_stack((x[:-1], x[1:]))
    ys = np.column_stack((y[:-1], y[1:]))
    v = results['velocity']
    segments = LineCollection([((xx[0], yy[0]), (xx[1], yy[1])) for xx, yy in zip(ys, xs)],
                              cmap=plt.cm.Greys, alpha=.6, clim=(0, 150), linewidths=.2)
    segments.set_array(v)
    ax.add_collection(segments)


def stars(pvalue):
    if pvalue < 0.001:
        return '***'
    elif pvalue < 0.01:
        return '**'
    elif pvalue < 0.05:
        return '*'
    else:
        return 'ns'
