#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""World Pendulum Alliance tides experiment (post processing).

This script reads and process the outputed data (csv files) generated
by the tides.py (unattended) script in order to calculate the average
pendulum period, corrected gravitational acceleration etc.

Author:   b g e n e t o @ g m a i l . c o m
History:  v1.0.0 Initial release
          v1.0.1 Added pendulum selection
          v1.0.2 Added export button for average csv file
          v1.0.3 Remove append function with concat+transpose
          v1.0.4 Added slider for changing CTE
          v1.0.5 Added script version in bottom of the page
          v1.0.6 Added num plots slider and copyright notice
          v1.0.7 Use 'data' directory to save and download files
          v1.0.8 Download files from password protected folder
          v1.1.0 Save raw_data to pickle file to improve performance
          v1.1.1 Correct counting number of csv files
          v1.1.2 Added trendline to specific plots
          v1.1.3 Using real fft version from scipy
          v1.1.4 Added slider for points in FFT
          v1.1.5 Added more historical plots
          v1.2.5 Deleted low noise removal procedure
                 Added error bars and std error computation
                 Using os.replace to implement atomic file write
                 Write all objects in one pickle file with pkl extension
          v1.2.6 Secondary pendulum selected by default
                 Added procedure to remove csv lines with spurious data (outliers)
          v1.2.7 Start DFT plot from 0 to 4 instead of 0.3 to 4. 
                 Change CTE slider to radio. Added function to find optimum cte value                  
          v1.2.8 Added interpolated fft plots
          v1.2.9 Select max number of points for fft plots

Usage:
    $ streamlit run tides-st.py
"""

import itertools
import math
from operator import rshift
import os
import pickle
import sys
import uuid
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed, parallel_backend
from plotly.subplots import make_subplots
from scipy import stats

import streamlit as st

__author__ = "Bernhard Enders"
__maintainer__ = "Bernhard Enders"
__email__ = "b g e n e t o @ g m a i l d o t c o m"
__copyright__ = "Copyright 2022, Bernhard Enders"
__license__ = "GPL"
__version__ = "1.2.9"
__status__ = "Development"
__date__ = "20220215"


def stop(code=0):
    if st._is_running_with_streamlit:
        st.stop()
    sys.exit(code)


def par_myavg_err(key: str, df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=0).rename(key), df.std(axis=0).rename(key)/math.sqrt(len(df.index))


def myavg(dfd: dict[str, pd.DataFrame], percentile: float = 0.0) -> pd.DataFrame:
    """Compute the (trimmed) mean of a dictionary of DataFrames

    Args:
        dfd (pd.DataFrame): dictionary of DataFrames
        percentile (float, optional): percentile to trim in both extremes. Defaults to 0.0.

    Returns:
        pd.DataFrame: the (trimmed) mean (or std deviation) DataFrame
    """
    df = pd.DataFrame()
    ndf = pd.DataFrame()
    for key in dfd:
        # remove the first column (samples) from the DataFrame
        df = dfd[key].drop(dfd[key].columns[0], axis=1)

        # (optional) remove outliers from the DataFrame
        if percentile != 0.0:
            cols = df.columns
            df = pd.DataFrame(stats.trimboth(df, percentile))
            df.columns = cols

        ps = df.mean(axis=0)
        # ndf = ndf.append(ps.rename(key))  # append method is deprecated
        ndf = pd.concat([ndf, ps.rename(key)], axis=1)

    return ndf.transpose().sort_index()


def mystdev(dfd: dict[str, pd.DataFrame], percentile: float = 0.0) -> pd.DataFrame:
    """Compute the (trimmed) standard deviation of a dictionary of DataFrames

    Args:
        dfd (pd.DataFrame): dictionary of DataFrames
        percentile (float, optional): percentile to trim in both extremes. Defaults to 0.0.

    Returns:
        pd.DataFrame: the (trimmed) mean DataFrame
    """
    df = pd.DataFrame()
    ndf = pd.DataFrame()
    for key in dfd:
        # remove the first column (samples) from the DataFrame
        df = dfd[key].drop(dfd[key].columns[0], axis=1)

        # remove outliers from the DataFrame if requested
        if percentile != 0.0:
            cols = df.columns
            df = pd.DataFrame(stats.trimboth(df, percentile))
            df.columns = cols

        ps = df.std(axis=0)
        # ndf = ndf.append(ps.rename(key))  # append method is deprecated
        ndf = pd.concat([ndf, ps.rename(key)], axis=1)

    return ndf.transpose().sort_index()


def par_mystderr(key: str, df: pd.DataFrame) -> pd.Series:
    # remove the first column (samples) from the DataFrame
    df = df.drop(df.columns[0], axis=1)
    ps = df.std(axis=0)/math.sqrt(len(df.index))
    return ps


def mystderr(dfd: dict[str, pd.DataFrame], percentile: float = 0.0) -> pd.DataFrame:
    """Compute the (trimmed) standard error of a dictionary of DataFrames

    Args:
        dfd (pd.DataFrame): dictionary of DataFrames
        percentile (float, optional): percentile to trim in both extremes. Defaults to 0.0.

    Returns:
        pd.DataFrame: the (trimmed) std error DataFrame
    """
    df = pd.DataFrame()
    ndf = pd.DataFrame()
    for key in dfd:
        # remove the first column (samples) from the DataFrame
        df = dfd[key].drop(dfd[key].columns[0], axis=1)

        # remove outliers from the DataFrame if requested
        if percentile != 0.0:
            cols = df.columns
            df = pd.DataFrame(stats.trimboth(df, percentile))
            df.columns = cols

        ps = df.std(axis=0)/math.sqrt(len(df.index))
        # ndf = ndf.append(ps.rename(key))  # append method is deprecated
        ndf = pd.concat([ndf, ps.rename(key)], axis=1)

    return ndf.transpose().sort_index()


def filename_only(fn):
    """Returns the filename without path and (the last) file extension.

    Args:
        fn: fullpath to file.

    Returns:
        The filename fn without both path and the last extension.

    """
    return os.path.splitext(os.path.basename(fn))[0]


def check_missing_lines(raw_data: dict[str, pd.DataFrame],
                        num_csv_files: int,
                        archive) -> None:
    '''Ignore/remove files with insufficient lines and lines
       with spurious data.
    '''
    # discard if moving average greater than period threshold
    threshold = 0.005

    # min number of lines in csv files to be considered as valid
    min_lines_csv = 10 if csv.lines/2 > 10 else csv.lines/2
    for key, df in raw_data.items():
        nlines = len(df)
        sr = df['period'].rolling(3).median()
        difference = np.abs(df['period'] - sr)
        outlier_idx = difference > threshold
        num_outliers = len(df['period'][outlier_idx])
        if num_outliers > 0:
            # remove outliers and reset index
            raw_data[key].drop(df[outlier_idx].index, inplace=True)
            raw_data[key].reset_index(drop=True, inplace=True)
            st.warning(
                f":warning: Removing {num_outliers} outlier(s) line(s) in file {key}.csv")
        if nlines < min_lines_csv:
            st.warning(
                f":warning: Ignoring file {key}. Wrong number of lines.")
            csv.missing_files.append(key)
            del raw_data[key]

    if len(raw_data) + len(csv.missing_files) == num_csv_files:
        st.success(
            f":white_check_mark: {num_csv_files} files successfully processed!")
    else:
        try:
            os.remove(archive)
        except:
            pass
        st.error(
            ":x: Not all csv files were read successfully. Please refresh (F5) this page!")
        stop()


def par_load_data(f):
    """Read all CSV files and check for missing lines or lines outside
       the expected values. We put all raw data from csv files into
       a dictionary of pandas DataFrame.
    """
    try:
        df = pd.read_csv(f.resolve(), names=csv.colnames, skiprows=1,
                         header=None, float_precision='round_trip')
    except:
        st.error(f":x: Error reading csv file named {f.stem}")

    return f.stem, df


def load_data(all_files: list) -> dict:
    """Read all CSV files and check for missing lines or lines outside
       the expected values. We put all raw data from csv files into
       a dictionary of pandas DataFrame.
    """
    # read all files
    raw_data = {}
    for f in all_files:
        try:
            raw_data[f.stem] = pd.read_csv(f.resolve(), names=csv.colnames, skiprows=1,
                                           header=None, float_precision='round_trip')
        except:
            st.error(f":x: Error reading csv file named {f.stem}")

    # sort by date/filename
    ret = dict(sorted(raw_data.items()))

    return ret


def check_missing_files(raw_data: dict[str, pd.DataFrame]) -> None:
    """Check missing files
    We check if there are any missing csv files.
    In other words, check if we missed any pendulum runs.
    """
    try:
        d1 = datetime.strptime(list(raw_data)[0], csv.date_format)
    except Exception as e:
        display.fatal(
            "Wrong date format! Please check provided csv date format.")
        stop()

    wstr = ''
    for dtime in raw_data:
        d2 = datetime.strptime(dtime, csv.date_format)
        dt = (d2 - d1).total_seconds()/60.0
        n = math.floor(dt/exp.dt) - 1
        if dt > exp.dt and n > 0:
            wstr += f":warning: You have **{n} missing** data file(s) before **{dtime}**  \n"
            for m in range(1, n+1):
                mfile = (
                    d2 - timedelta(minutes=exp.dt*m)
                ).strftime(csv.date_format)
                csv.missing_files.append(mfile)
        # update date
        d1 = d2

    st.warning(wstr)


def cte_optimization(cte: float, raw_data: dict[str, pd.DataFrame]) -> int:
    ndf = pd.DataFrame()
    for key, df in raw_data.items():
        ndf.loc[key, 'length'] = (pdl.length *
                                  (1 + cte*(df['temperature'] - pdl.temperature))).mean()
        ndf.loc[key, 'gravity'] = (4 *
                                   math.pi**2*ndf.loc[key, 'length']/df['period']**2).mean()
    # compute variance
    variance = ndf['gravity'].var()
    return (cte, variance)


def par_thermal_correction(key: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    ret = pd.DataFrame(index=df.index)
    if pdl.temperature == 0.0:
        # use only first points to compute mean temperature
        # thermistor inside abs enclosure issue
        temperature_c = df['temperature'][0:6].mean()
        ret['length_c'] = pdl.length
    else:
        temperature_c = df['temperature'].mean()
        ret['length_c'] = pdl.length * \
            (1 + pdl.cte*(df['temperature'] - pdl.temperature))
    # uncorredted gravity
    ret['gravity'] = 4*math.pi**2*pdl.length/df['period']**2
    # correction by length expansion/contraction
    ret['gravity_c2'] = 4 * \
        math.pi**2*ret['length_c']/df['period']**2
    ret['gravity_c'] = 4 * \
        math.pi**2*pdl.length * \
        (1 + pdl.cte*(temperature_c -
                      df['temperature']))/df['period']**2

    return key, ret, temperature_c


def thermal_correction(raw_data: dict[str, pd.DataFrame]) -> pd.Series:
    temperature_c = pd.Series(dtype='float64')
    for key, df in raw_data.items():
        if pdl.temperature == 0.0:
            # use only first points to compute mean temperature
            # thermistor inside abs enclosure issue
            temperature_c[key] = df['temperature'][0:6].mean()
            raw_data[key]['length_c'] = pdl.length
        else:
            temperature_c[key] = df['temperature'].mean()
            raw_data[key]['length_c'] = pdl.length * \
                (1 + pdl.cte*(df['temperature'] - pdl.temperature))
        # uncorredted gravity
        raw_data[key]['gravity'] = 4*math.pi**2*pdl.length/df['period']**2
        # correction by length expansion/contraction
        raw_data[key]['gravity_c2'] = 4 * \
            math.pi**2*df['length_c']/df['period']**2
        raw_data[key]['gravity_c'] = 4 * \
            math.pi**2*pdl.length * \
            (1 + pdl.cte*(temperature_c[key] -
                          df['temperature']))/df['period']**2

    return temperature_c


class Pendulum:
    __slots__ = 'length', 'temperature', 'cte'

    def __init__(self,
                 length: float = 0.0,
                 temperature: float = 0.0,
                 cte: float = 14e-6) -> None:
        # measured pendulum length (including ball radius)
        self.length = length
        # temperature at measured length
        self.temperature = temperature
        # pendulum wire coefficient of thermal expansion
        self.cte: cte


class Experiment:
    def __init__(self, dx: float, dt: float, osc: int) -> None:
        # pendulum displacement in cm
        self.dx = dx
        # delta-time between each run in minutes
        self.dt = dt
        # number of oscilations before stop
        self.osc = osc


class Output:
    """Style our output"""

    def __init__(self, function: Callable,
                 heading: str = "###",
                 use_emoji: bool = False) -> None:
        self.heading = heading
        self.function = function
        self.use_emoji = use_emoji
        self.emoji: str = ""
        self.prefix: str = ""

    def msg(self, msg: str, heading: str = None) -> None:
        if not self.use_emoji:
            self.emoji = ""
        if not heading:
            heading = self.heading
        msg = str(heading + self.emoji + self.prefix + str(msg)).strip()
        self.function(msg)

    def fatal(self, msg: str) -> None:
        self.emoji = " :red_circle: "
        self.prefix = " FATAL: "
        self.msg(msg)

    def debug(self, msg: str) -> None:
        self.emoji = " :bug: "
        self.prefix = " DEBUG: "
        self.msg(msg)

    def error(self, msg: str) -> None:
        self.emoji = " :x: "
        self.prefix = " ERROR: "
        self.msg(msg)

    def warning(self, msg: str) -> None:
        self.emoji = " :warning: "
        self.prefix = " WARNING: "
        heading = self.heading + "##"
        self.msg(msg, heading)

    def info(self, msg: str) -> None:
        self.emoji = " ⓘ "
        self.prefix = " INFO: "
        self.msg(msg)

    def success(self, msg: str) -> None:
        self.emoji = " :white_check_mark: "
        self.prefix = ""
        self.msg(msg)

    def check(self, msg: str) -> None:
        self.emoji = " :ballot_box_with_check: "
        self.prefix = ""
        self.msg(msg)

    def wait(self, msg: str) -> None:
        self.emoji = " :stopwatch: "
        self.prefix = ""
        self.msg(msg)

    def print(self, msg) -> None:
        self.emoji = ""
        self.prefix = ""
        self.function(msg)


def st_layout(title: str = "Streamlit App") -> None:
    """Configure Streamlit page layout"""

    st.set_page_config(
        page_title=title.split('-')[0],
        page_icon=":ocean:",
        layout="wide")
    st.title(title)
    # hide main (top right) menu
    hide_menu_style = """
        <style>
        # MainMenu {visibility: hidden;}
        </style>
        """
    # st.markdown(hide_menu_style, unsafe_allow_html=True)


def download_archive(fn: str, output_dir: str) -> bool:
    """Download file from Nextcloud"""
    display.wait("Downloading pendulum archive data...")
    import requests
    fpath = os.path.join(output_dir, fn)
    share_id = 'kbiz4qzDdkFAygb'
    share_passwd = 'ij0ndzX6'
    auth = (share_id, share_passwd)
    headers = {"X-Requested-With": "XMLHttpRequest"}
    url = f"https://cloud.bgeneto.com.br:4433/public.php/webdav/{fn}"
    if not share_passwd:
        url = f"https://cloud.bgeneto.com.br:4433/s/{share_id}/download?path=%2F&files={fn}"
    try:
        if not share_passwd:
            response = requests.get(url, stream=True)
        else:
            response = requests.get(
                url, headers=headers, auth=auth, stream=True)
        if response.status_code == 200:
            with open(fpath, 'wb') as f:
                f.write(response.raw.read())
        else:
            return False
    except:
        return False

    return True


def extract_archive(fn: str, output_dir: str) -> bool:
    display.wait("Extracting archive data...")
    import tarfile
    fpath = os.path.join(output_dir, fn)
    destination_dir = os.path.join(output_dir, filename_only(fn))
    try:
        tar = tarfile.open(fpath, "r:gz")
        tar.extractall(destination_dir)
        tar.close()
    except:
        return False

    return True


def exp_avg_data(csvfn: str, avg_data: pd.DataFrame, compress: bool = False) -> str:
    exp_avg = avg_data.copy()

    # add zero-valued missing files
    for mfile in csv.missing_files:
        exp_avg.loc[mfile] = 0.0

    # reorder indexes after adding missing
    exp_avg.sort_index(inplace=True)

    # setup compression
    compression_opts = None
    fext = '.csv'
    if compress:
        compression_opts = dict(method='zip',
                                archive_name=csvfn+fext)
        fext = '.zip'

    # save csv file
    exp_avg.to_csv(csvfn+fext, index=True,
                   compression=compression_opts)

    return csvfn+fext


def archive_too_old(fn: str, output_dir: str) -> bool:
    """Check if file is old enough to be downloaded again"""
    fpath = os.path.join(output_dir, fn)
    if not os.path.isfile(fpath):
        return True
    import time

    # 20 minutes old to (downloaded archive) be considered old
    min_delta = 20*60
    # get file modification time
    ftime = os.path.getmtime(fpath)
    # current time
    now = time.time()
    if (now - ftime) > min_delta:
        return True

    return False


def cte_status():
    st.session_state.cte_changed = True
    index = st.session_state.cte_radio
    st.session_state.cte = exp.cte[index]


def initial_sidebar_config():
    # sidebar contents
    sidebar = st.sidebar
    sidebar.subheader("..:: MENU ::..")
    # pendulum radio box
    sidebar.radio(
        "Please choose a pendulum:",
        ('UnB Secondary', 'UnB Primary'),
        index=0,
        key="pendulum",
        on_change=cte_status)
    # date selector
    sidebar.date_input("Choose a day to plot:", key="plot_date")
    # default session values
    if 'cte_changed' not in st.session_state:
        st.session_state.cte_changed = False
        st.session_state.cte = 14
    # CTE slider
    # sidebar.slider('Wire CTE ⨯ 10⁻⁶ (default = 14):',
    #               1,
    #               100,
    #               step=1,
    #               value=14,
    #               key="cte",
    #               on_change=cte_status)
    sidebar.radio(
        "Wire CTE value:",
        ('default', f'optimized'),
        index=0,
        key="cte_radio",
        on_change=cte_status)

    # number of individual plots slider
    sidebar.slider('How many individual plots?',
                   1,
                   10,
                   value=1,
                   step=1,
                   key="nlast")

    return sidebar


def historical_plots(options, avg_data):
    # plot without indexes
    df = avg_data.reset_index(drop=True)

    # period vs temperature
    # create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df['period'], name="period"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['temperature'], name="temperature"),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="PERIOD VS TEMPERATURE (averaged - historical)"
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="period (s)", secondary_y=False)
    fig.update_yaxes(title_text="temperature (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # period vs corrected gravity
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df['period'], name="period"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['gravity_c2'], name="gravity_c2"),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="PERIOD VS GRAVITY (averaged - historical)"
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="period (s)", secondary_y=False)
    fig.update_yaxes(title_text="gravity_c2 (m/s²)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # temperature vs corrected gravity
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df['temperature'], name="temperature"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['gravity_c2'], name="gravity_c2"),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="TEMPERATURE VS GRAVITY (averaged - historical)"
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="gravity_c2 (m/s²)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # corrected length vs corrected gravity
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df['length_c'], name="length_c"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['gravity_c2'], name="gravity_c2"),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="length_c x gravity_c2 (averaged - historical)"
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="length_c", secondary_y=False)
    fig.update_yaxes(title_text="gravity_c2", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # plot gravity comparison
    fig = px.line(df,
                  y=['gravity', 'gravity_c', 'gravity_c2'],
                  x=df.index,
                  title="GRAVITY (averaged - historical)",
                  labels={"index": "run", "value": "gravity"})
    # fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig, use_container_width=True)

    # plot selected data
    for col in options:
        sr = avg_data[col].reset_index(drop=True)
        fig = px.scatter(sr,
                         y=sr.values,
                         x=sr.index,
                         title=f"{col.upper()} (average - historical)",
                         labels={"index": "run", "y": sr.name})
        st.plotly_chart(fig, use_container_width=True)


def individual_plots(options, plot_date, nlast, raw_data):
    nraw_data = {k: v for k, v in raw_data.items() if k.startswith(plot_date)}
    for col in options:
        tl = None
        mode = 'markers+lines'
        if col in ['temperature', 'velocity']:
            tl = 'ols'
            mode = 'markers'
        for i in range(1, nlast+1):
            try:
                key = list(nraw_data)[-1*i]
            except:
                break
            if col in nraw_data[key].columns:
                sr = nraw_data[key][col]
                lsr = len(sr)
                avg = sr.mean()
                stdev = sr.std()
                err = stdev/math.sqrt(lsr)
                error_y = None if col in [
                    'temperature', 'velocity'] else [err]*lsr
                fig = px.scatter(sr,
                                 y=sr.values,
                                 x=sr.index,
                                 title=f"{col.upper()} @ {key} [{lsr} pts | avg: {avg:.7f} | err: {err:.3e}]",
                                 labels={"index": "oscillation", "y": sr.name},
                                 trendline=tl,
                                 error_y=error_y)
                fig.data[0].update(mode=mode)
                st.plotly_chart(fig, use_container_width=True)


def daily_plots_old(options, plot_date, avg_data, stderr_data):
    '''daily plots with error bars'''
    for col in options:
        df = avg_data[col]
        sr = df.filter(like=plot_date)
        esr = stderr_data[col].filter(like=plot_date)
        ldf = len(sr)
        if ldf > 0:
            avg = sr.mean()
            std = sr.std()
            fig = px.scatter(sr,
                             y=sr.values,
                             x=sr.index,
                             title=f"{col.upper()} @ {plot_date} [{ldf} runs | avg: {avg:.7f} | stdev: {std:.3e}]",
                             labels={"index": "run", "y": sr.name},
                             error_y=esr)
            fig.update_traces(mode='markers+lines')
            st.plotly_chart(fig, use_container_width=True)
        else:
            display.warning("No data available in the selected day!")


def daily_plots(options, plot_date, avg_data, stderr_data):
    '''daily plots with continuous error bands'''
    for col in options:
        df = avg_data[col]
        sr = df.filter(like=plot_date)
        # std error upper and lower bounds
        uerr = sr+stderr_data[col].filter(like=plot_date)
        lerr = sr-stderr_data[col].filter(like=plot_date)
        ldf = len(sr)
        if ldf > 0:
            avg = sr.mean()
            std = sr.std()
            fig = go.Figure([
                go.Scatter(
                    name=col,
                    y=sr.values,
                    x=sr.index,
                    mode='markers+lines',
                    showlegend=False
                ),
                go.Scatter(
                    name='upper bound',
                    y=uerr,
                    x=sr.index,
                    mode='lines',
                    marker=dict(color="#224eb5"),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='lower bound',
                    y=lerr,
                    x=sr.index,
                    marker=dict(color="#224eb5"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(34, 78, 181, 0.3)',
                    fill='tonexty',
                    showlegend=False
                )
            ])
            fig.update_layout(
                title=f"{col.upper()} @ {plot_date} [{ldf} runs | avg: {avg:.7f} | stdev: {std:.3e}]",
                hovermode="x")
            st.plotly_chart(fig, use_container_width=True)
        else:
            display.warning("No data available in the selected day!")


def interpolated_fft_plots(signal, npts: int = 0):
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import find_peaks

    col = signal.name
    avg = signal.mean()
    s = (signal-avg)
    s = s[0:npts]
    mult = 2
    idx = range(0, mult*len(s.index))
    sr = pd.Series(index=idx, name=col)
    sr.iloc[::2] = s.iloc[::1]
    sr = sr.interpolate(method='spline', order=3)
    idx = range(0, mult*len(sr.index))
    nsr = pd.Series(index=idx, name=col)
    nsr.iloc[::2] = sr.iloc[::1]
    nsr = nsr.interpolate(method='spline', order=3)
    nsr = nsr.to_numpy()

    # using scipy fft: it is faster
    yf = rfft(nsr)
    N = len(nsr)
    # time scale in days
    day = 24*60.0  # min
    dt = float(exp.dt)/mult/mult/day  # dt in unit of days
    t = np.linspace(0, N*dt, N, endpoint=False)
    xf = rfftfreq(N, dt)

    # normalized amplitude/power:
    amp = np.abs(2.0/N*yf)

    # plot the results
    fig = px.bar(x=xf,
                 y=amp,
                 labels={"x": "frequency (1/day)", "y": "amplitude"},
                 title=f"Interpolated DFT of {col.upper()} ({N} points used)")
    fig.update_yaxes(range=[0, 200e-6])
    fig.update_traces(marker_color='red')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'hovermode': "x"
    })
    st.plotly_chart(fig, use_container_width=True)

    xmask = np.where((xf >= 0.0) & (xf < 0.3))
    fig = px.bar(x=xf[xmask],
                 y=amp[xmask],
                 labels={"x": "frequency (1/day)", "y": "amplitude"},
                 title=f"(0-0.3) Interpolated DFT of {col.upper()} ({N} points used)")
    fig.update_traces(marker_color='red')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'hovermode': "x"
    })
    st.plotly_chart(fig, use_container_width=True)

    xmask = np.where((xf >= 0.0) & (xf < 4.0))
    fig = px.bar(x=xf[xmask],
                 y=amp[xmask],
                 labels={"x": "frequency (1/day)", "y": "amplitude"},
                 title=f"(0.0-4) Interpolated DFT of {col.upper()} ({N} points used)")
    fig.update_traces(marker_color='red')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'hovermode': "x"
    })
    st.plotly_chart(fig, use_container_width=True)


def fft_plots(signal, npts: int = 0):
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import find_peaks

    npts = len(signal) if npts == 0 else npts
    col = signal.name
    avg = signal.mean()
    s = (signal-avg).to_numpy()
    s = s[0:npts]

    # using scipy fft: it is faster
    yf = rfft(s)
    N = len(s)
    # time scale in days
    day = 24*60.0  # min
    dt = float(exp.dt)/day  # dt in unit of days
    t = np.linspace(0, N*dt, N, endpoint=False)
    xf = rfftfreq(N, dt)

    # normalized amplitude/power:
    amp = np.abs(2.0/N*yf)

    # do not remove low frequency noise
    if False:
        num_hfn = 6
        xf = xf[num_hfn:]
        amp = amp[num_hfn:]

    # plot the results
    fig = px.bar(x=xf,
                 y=amp,
                 labels={"x": "frequency (1/day)", "y": "amplitude"},
                 title=f"DFT of {col.upper()} ({N} points used)")
    # fig.update_xaxes(range=[0, 2])
    st.plotly_chart(fig, use_container_width=True)

    xmask = np.where((xf >= 0.0) & (xf < 0.3))
    fig = px.bar(x=xf[xmask],
                 y=amp[xmask],
                 labels={"x": "frequency (1/day)", "y": "amplitude"},
                 title=f"(0-0.3) DFT of {col.upper()} ({N} points used)")
    # fig.update_xaxes(range=[0, 0.3])
    st.plotly_chart(fig, use_container_width=True)

    xmask = np.where((xf >= 0.0) & (xf < 4.0))
    fig = px.bar(x=xf[xmask],
                 y=amp[xmask],
                 labels={"x": "frequency (1/day)", "y": "amplitude"},
                 title=f"(0.0-4) DFT of {col.upper()} ({N} points used)")
    # fig.update_xaxes(range=[0.0, 4])
    st.plotly_chart(fig, use_container_width=True)

    # find peaks (not working, is missing first n peaks)
    # min_freq = 0.01
    # max_freq = 4
    # pos_mask = np.where((xf > min_freq) & (xf < max_freq))
    # freqs = xf[pos_mask]
    # amps = amp[pos_mask]
    # peaks, _ = find_peaks(amps, height=0)

    # fig = px.bar(x=freqs[peaks],
    #              y=amps[peaks],
    #              labels={"x": "frequency (1/day)", "y": "amplitude"},
    #              title=f"DFT Peaks ({npts} points used)")
    # st.plotly_chart(fig, use_container_width=True)


def extra_plots(df):
    # temperature vs corrected gravity
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df['length_c'], name="length_c"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['gravity_c2'], name="gravity_c2"),
        secondary_y=True,
    )
    fig.update_layout(
        title_text="length_c x gravity_c2 (averaged - historical)"
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="length_c", secondary_y=False)
    fig.update_yaxes(title_text="gravity_c2", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


def main():
    start = timer()

    # optimum cte value (run cte_optimization to calculate)
    exp.cte = {'default': 14, 'optimized': 40.59375}

    sidebar = initial_sidebar_config()

    # selected wire cte from widget
    pdl.cte = 1e-6*st.session_state.cte

    # pendulum data according to the location
    archive = None
    p = st.session_state.pendulum
    if p == 'UnB Primary':
        archive = 'tides-data-WP-BSB.tgz'
        pdl.length = 2.8676
        pdl.temperature = 0.0
    elif p == 'UnB Secondary':
        pdl.length = 2.62135
        pdl.temperature = 22.6260
        archive = 'tides-data-wpafup.tgz'

    if not archive:
        display.fatal("Invalid pendulum selection!")
        stop()

    st.subheader(f"Showing :ocean: data from: **{p} Pendulum**")
    st.info(
        f":bulb: Using CTE = {pdl.cte:.4e}")

    # create output directory
    output_dir = 'data'

    # directory containing all csv files
    input_dir = os.path.join(output_dir, filename_only(archive))

    if not os.path.isdir(input_dir):
        try:
            os.makedirs(input_dir)
        except:
            display.fatal("Cannot create output directory!")
            stop()

    # cache files per pendulum
    cache_file = input_dir+'.pkl'
    no_cache_file = False
    if not os.path.isfile(cache_file):
        no_cache_file = True
        # required since os.replace gives error if file not exists
        Path(cache_file).touch()

    # count current number of csv in input_dir
    all_files = [f for f in Path(input_dir).glob('*.csv')]
    num_csv_files = len(all_files)

    # avoid multiple downloads from widget changes
    raw_data = {}
    avg_data = pd.DataFrame()
    stderr_data = pd.DataFrame()
    cache_miss = archive_too_old(
        archive, output_dir) or no_cache_file
    if cache_miss:
        # download compressed archive
        if not download_archive(archive, output_dir):
            display.fatal(
                "Cannot download compressed archive from remote location!")
            st.warning("Try again by reloading this page (F5).")
            stop()

        # extract downloaded file
        if not extract_archive(archive, output_dir):
            display.fatal("Error extracting the compressed archive!")
            stop()

        # after extracting, we list of all csv in input_dir for reading
        all_files = [f for f in Path(input_dir).glob('*.csv')]
        num_csv_files = len(all_files)

        placeholder = st.empty()
        placeholder.write(
            "#### :stopwatch: Please wait while parsing csv files...")

        # load/read csv files
        with st.spinner('Wait for it...'):
            if not parallel_computing:
                raw_data = load_data(all_files)
            else:
                with parallel_backend('multiprocessing', n_jobs=-2):
                    results = Parallel()(delayed(par_load_data)(f)
                                         for f in all_files)
                # join data
                for key, df in results:
                    raw_data[key] = df
                # sort data
                raw_data = dict(sorted(raw_data.items()))
        placeholder.empty()
    else:
        # load cached raw_data (deserialize)
        try:
            with open(cache_file, 'rb') as handle:
                raw_data = pickle.load(handle)
                avg_data = pickle.load(handle)
                stderr_data = pickle.load(handle)
        except:
            st.error(
                ":x: Corrupted cached data! Please refresh (F5) this page.")
            os.remove(cache_file)
            stop()

    # check missing lines then missing csv files
    check_missing_lines(raw_data, num_csv_files,
                        os.path.join(output_dir, archive))
    check_missing_files(raw_data)

    if compute_best_cte:
        with st.spinner('Computing best CTE value...'):
            arg = dict(itertools.islice(raw_data.items(), 1060, 1501))
            results = Parallel(n_jobs=-2)(delayed(cte_optimization)(x, arg)
                                          for x in 1.e-6*np.arange(14.0, 45.0, 0.03125))
            best_cte = min(results, key=lambda t: t[1])[0]
        st.subheader(f"best_cte = {best_cte}")

    # cte changed value, recompute
    if st.session_state.cte_changed or cache_miss:
        t0 = timer()
        with st.spinner('Wait, computing average, error and corrected values...'):
            # thermal expansion correction
            if not parallel_computing:
                avg_data['temperature_c'] = thermal_correction(raw_data)
            else:
                with parallel_backend('multiprocessing', n_jobs=-2):
                    results = Parallel()(delayed(par_thermal_correction)(
                        key, df) for key, df in raw_data.items())
                for key, df, tc in results:
                    raw_data[key] = pd.concat([raw_data[key], df], axis=1)
                    avg_data.loc[key, 'temperature_c'] = tc

            # compute std dev and average data
            avg_data = myavg(raw_data)
            stderr_data = mystderr(raw_data)
            tf = timer()
            st.caption(f"Computation time: {tf-t0:.2f}")

        # cache loaded data to disk (via pickle) using temp file (safer)
        tmpfn = output_dir + os.sep + str(uuid.uuid4().hex)
        try:
            with open(tmpfn, 'wb') as handle:
                pickle.dump(raw_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(avg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(stderr_data, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmpfn, cache_file)  # safer: atomic operation
        except:
            os.remove(tmpfn)
    st.info(
        ":bulb: *Local timezone is UTC–3 but all timestamps are displayed in UTC.*")

    # choose proper gravity correction based on available initial temperature
    gravity = 'gravity_c' if pdl.temperature == 0 else 'gravity_c2'
    default_options = ['period', gravity, 'temperature', 'velocity']

    # add data selection to sidebar
    options = sidebar.multiselect(
        label='Which data to plot?',
        options=avg_data.columns,
        default=default_options)

    # add fft points to sidebar
    mpts = 128
    fftpts = sidebar.slider('How many points to use in FFT?',
                            2*mpts,
                            len(avg_data[gravity]),
                            value=len(avg_data[gravity]),
                            step=mpts,
                            key="fftpts")

    st.success(":point_down: Aggregated data available to download below")
    display.print(
        "Click the button below :arrow_down: to download up-to-date **averaged** data file")
    # export all average data to CSV
    csvfn = exp_avg_data(f"{input_dir}-avg", avg_data)

    # create download button
    with open(csvfn) as f:
        st.download_button('Download CSV', f,
                           file_name=os.path.basename(csvfn), mime='text/csv')

    with st.expander("..:: SAMPLE DATA ::.."):
        # print some data (from last experiment run)
        last_key = list(raw_data)[-1]
        st.subheader("Raw data from last run")
        st.write(raw_data[last_key].iloc[:, 1:].style.format("{:.7}"))

        st.subheader("Averaged data stats")
        st.write(avg_data.describe().style.format("{:.7}"))

        # min and max values from averaged data
        mean_min = avg_data.min(axis=0)
        mean_max = avg_data.max(axis=0)

        st.subheader("Delta (max-min) from averaged data")
        st.write(
            (mean_max - mean_min).rename("delta").to_frame().style.format("{:.7}"))

    # choosen plot data
    nlast = st.session_state.nlast
    plot_date = str(st.session_state.plot_date)

    # historical plots
    with st.expander("..:: HISTORICAL PLOTS ::.."):
        st.subheader("HISTORICAL PLOTS")
        historical_plots(options, avg_data)

    # individual plots
    with st.expander("..:: INDIVIDUAL PLOTS ::.."):
        st.subheader("INDIVIDUAL PLOTS")
        st.info(":point_left: Please select the **desired date** \
                and the **number of plots** (*n*) in the left menu")
        st.success(":bulb: This will show only *n* last runs from the \
                   selected date choosen on the left menu")
        individual_plots(options, plot_date, nlast, raw_data)

    # daily plots
    with st.expander("..:: DAILY PLOTS ::.."):
        st.subheader("DAILY PLOTS")
        st.info(":point_left: Please select the **desired date** in the left menu")
        daily_plots(options, plot_date, avg_data, stderr_data)

    # fft plot in fftpts from slider
    fft_plots(avg_data[gravity], fftpts)

    interpolated_fft_plots(avg_data[gravity], fftpts)

    # extra_plots(avg_data)

    # copyright, version and running time info
    end = timer()
    st.caption(
        f":copyright: 2022 bgeneto | Version: {__version__} | Execution: {(end-start):.2f}s")

    # reset cte change status
    st.session_state.cte_changed = False


if __name__ == '__main__':
    # always run as a streamlit app
    force_streamlit = True

    # compute best cte only once and use the result
    compute_best_cte = False

    # enable parallel processing
    parallel_computing = True

    # page title/header
    title = "WPA Tides Experiment - Realtime Data Visualization"

    # pendulum data
    pdl = Pendulum()

    # experiment data
    exp = Experiment(dx=15, dt=23, osc=64)

    # csv anonymous class
    csv = SimpleNamespace(lines=exp.osc,
                          date_format="%Y-%m-%dT%Hh%M",
                          missing_files=[])

    # rename columns in csv files
    csv.colnames = ['sample', 'period',
                    'gravity_pic', 'velocity', 'temperature']

    # increase pandas default output precision from 6 decimal places to 7
    pd.set_option("display.precision", 7)

    # configure print output (streamlit, python, ipython etc...)
    display = Output(st.write, "####", True)

    # check if running as standalone python script or via streamlit
    if st._is_running_with_streamlit:
        st_layout(title)
        main()
    else:
        if force_streamlit:
            st_layout(title)
            from streamlit import cli as stcli
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
        else:
            display = Output(print, "")
            main()
