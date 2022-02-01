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
          v1.0.9 Save raw_data to pickle file to improve performance
Usage:
    $ streamlit run tides-st.py
"""

import base64
import math
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from re import S
from timeit import default_timer as timer
from types import SimpleNamespace
from typing import Callable

import pandas as pd
import plotly.express as px
from scipy import stats

import streamlit as st

__author__ = "Bernhard Enders"
__maintainer__ = "Bernhard Enders"
__email__ = "b g e n e t o @ g m a i l d o t c o m"
__copyright__ = "Copyright 2022, Bernhard Enders"
__license__ = "GPL"
__version__ = "1.0.9"
__status__ = "Development"
__date__ = "20220201"


def stop(code=0):
    if st._is_running_with_streamlit:
        st.stop()
    sys.exit(code)


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
        pd.DataFrame: the (trimmed) mean (or std deviation) DataFrame
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


def filename_only(fn):
    """Returns the filename without path and (the last) file extension.

    Args:
        fn: fullpath to file.

    Returns:
        The filename fn without both path and the last extension.

    """
    return os.path.splitext(os.path.basename(fn))[0]


def check_missing_lines(raw_data: dict[str, pd.DataFrame], num_csv_files: int) -> None:
    # min number of lines in csv files to be considered as valid
    # interrupted run may occur
    min_lines_csv = 10 if csv.lines/2 > 10 else csv.lines/2
    for key, df in raw_data.items():
        nlines = len(df)
        if nlines < min_lines_csv:
            display.warning(
                f"Ignoring file {key}. Wrong number of lines.")
            csv.missing_files.append.append(key)
            del raw_data[key]

    if len(raw_data) + len(csv.missing_files) == num_csv_files:
        display.success(f"{num_csv_files} files successfully processed!")
    else:
        display.fatal(
            "Not all csv files were read successfully.")
        stop()


def load_data(all_files: list) -> dict:
    """Read all CSV files
    We put all raw data from csv files into a dictionary of pandas DataFrame.
    """
    # read all files
    raw_data = {}
    for f in all_files:
        try:
            raw_data[f.stem] = pd.read_csv(f.resolve(), names=csv.colnames, skiprows=1,
                                           header=None, float_precision='round_trip')
        except:
            pass

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

    for dtime in raw_data:
        d2 = datetime.strptime(dtime, csv.date_format)
        dt = (d2 - d1).total_seconds()/60.0
        n = math.floor(dt/exp.dt) - 1
        if dt > exp.dt and n > 0:
            display.warning(
                f"You have {n} missing data file(s) before {dtime}")
            for m in range(1, n+1):
                mfile = (
                    d2 - timedelta(minutes=23*m)
                ).strftime(csv.date_format)
                csv.missing_files.append(mfile)
        # update date
        d1 = d2


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
        raw_data[key]['gravity_c'] = 4*math.pi**2*pdl.length * \
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
        #MainMenu {visibility: hidden;}
        </style>
        """
    #st.markdown(hide_menu_style, unsafe_allow_html=True)


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


def get_table_download_link(df, fn):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv_file = df.to_csv(index=True)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv_file.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{fn}">download csv file</a>'
    return href


def exp_avg_data(csvfn: str, avg_data: pd.DataFrame, compress: bool = False) -> str:
    exp_avg = avg_data.copy()

    # add zero-valued missing files
    for mfile in csv.missing_files:
        exp_avg.loc[mfile] = 0

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


def initial_sidebar_config():
    # sidebar contents
    sidebar = st.sidebar
    sidebar.subheader("..:: MENU ::..")
    # pendulum radio box
    sidebar.radio(
        "Please choose a pendulum:",
        ('UnB - Primary', 'UnB - Secondary'),
        index=0,
        key="pendulum",
        on_change=cte_status)
    # date selector
    sidebar.date_input("Choose a day to plot:", key="plot_date")
    # CTE slider
    if 'cte_changed' not in st.session_state:
        st.session_state.cte_changed = False
    sidebar.slider('Wire CTE ⨯ 10⁻⁶ (default = 14):',
                   1,
                   100,
                   step=1,
                   value=14,
                   key="cte",
                   on_change=cte_status)
    # number of individual plots slider
    sidebar.slider('How many individual plots?',
                   1,
                   10,
                   value=3,
                   step=1,
                   key="nlast")

    return sidebar


def historical_plots(options, avg_data):
    # plot gravity comparison
    df = avg_data.reset_index(drop=True)
    fig = px.line(df,
                  y=['gravity', 'gravity_c', 'gravity_c2'],
                  x=df.index,
                  title="GRAVITY (averaged - historical)",
                  labels={"index": "run", "value": "gravity"})
    # fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig, use_container_width=True)

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
        for i in range(1, nlast+1):
            try:
                key = list(nraw_data)[-1*i]
            except:
                break
            if col in nraw_data[key].columns:
                sr = nraw_data[key][col]
                lsr = len(sr)
                stdev = sr.std()
                err = stdev/math.sqrt(lsr)
                fig = px.scatter(sr,
                                 y=sr.values,
                                 x=sr.index,
                                 title=f"{col.upper()} @ {key} [{lsr} pts | err: {err:.3e}]",
                                 labels={"index": "oscillation", "y": sr.name})
                fig.update_traces(mode='markers+lines')
                st.plotly_chart(fig, use_container_width=True)


def daily_plots(options, plot_date, avg_data):
    for col in options:
        df = avg_data[col]
        sr = df.filter(like=plot_date)
        ldf = len(sr)
        if ldf > 0:
            avg = sr.mean()
            std = sr.std()
            fig = px.scatter(sr,
                             y=sr.values,
                             x=sr.index,
                             title=f"{col.upper()} @ {plot_date} [{ldf} runs | avg: {avg:.7f} | stdev: {std:.3e}]",
                             labels={"index": "run", "y": sr.name})
            fig.update_traces(mode='markers+lines')
            st.plotly_chart(fig, use_container_width=True)
        else:
            display.warning("No data available in the selected day!")


def main():
    start = timer()

    sidebar = initial_sidebar_config()

    # wire cte from slider
    pdl.cte = 1e-6*st.session_state.cte

    # pendulum data according to the location
    archive = None
    p = st.session_state.pendulum
    if p == 'UnB - Primary':
        archive = 'tides-data-WP-BSB.tgz'
        pdl.length = 2.8676
        pdl.temperature = 0.0
    elif p == 'UnB - Secondary':
        pdl.length = 2.62135
        pdl.temperature = 22.6260
        archive = 'tides-data-wpafup.tgz'

    if not archive:
        display.fatal("Invalid pendulum selection!")
        stop()

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

    # list of all csv and cached files in input_dir
    all_files = [f for f in Path(input_dir).glob('*.csv')]

    cached_files = [f for f in Path(input_dir).glob('*.pickle')]

    # avoid multiple downloads from widget changes
    raw_data = {}
    avg_data = pd.DataFrame()
    cache_miss = archive_too_old(archive, output_dir) or len(cached_files) < 1
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

        placeholder = st.empty()
        placeholder.write(
            "#### :stopwatch: Please wait while parsing csv files...")

        # load/read csv files
        with st.spinner('Wait for it...'):
            raw_data = load_data(all_files)

        placeholder.empty()
    else:
        # load cached raw_data (deserialize)
        with open(input_dir+'/raw.pickle', 'rb') as handle:
            raw_data = pickle.load(handle)
        # load cached avg_data
        with open(input_dir+'/avg.pickle', 'rb') as handle:
            avg_data = pickle.load(handle)

    # check missing lines then missing csv files
    check_missing_lines(raw_data, len(all_files))
    check_missing_files(raw_data)

    # cte changed value, recompute
    if st.session_state.cte_changed or cache_miss:
        # thermal expansion correction
        temperature_c = thermal_correction(raw_data)

        # compute std dev and average data
        avg_data = myavg(raw_data)
        avg_data['temperature_c'] = temperature_c

        # cache loaded data to disk (via pickle)
        with open(input_dir+'/raw.pickle', 'wb') as handle:
            pickle.dump(raw_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(input_dir+'/avg.pickle', 'wb') as handle:
            pickle.dump(avg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    st.warning(
        "*Local timezone is UTC—3 but all timestamps are displayed in UTC.*")

    # choose proper gravity correction based on available initial temperature
    gravity = 'gravity_c' if pdl.temperature == 0 else 'gravity_c2'
    default_options = ['period', gravity, 'temperature', 'velocity']

    # add to sidebar
    options = sidebar.multiselect(
        label='Which data to plot?',
        options=avg_data.columns,
        default=default_options)

    st.info("#### :point_down: Aggregated data available to download")
    display.print(
        "Click the button :arrow_down: below to download up-to-date **averaged** data file")
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
        daily_plots(options, plot_date, avg_data)

    # copyright, version and running time info
    end = timer()
    st.caption(
        f":copyright: 2022 bgeneto | Version: {__version__} | Execution time: {(end-start):.2f}")

    # reset cte change status
    st.session_state.cte_changed = False


if __name__ == '__main__':
    # always run as a streamlit app
    force_streamlit = False

    # page title/header
    title = "UnB WPA Tides Experiment - Realtime Data Visualization"

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
