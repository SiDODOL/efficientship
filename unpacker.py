import pandas as pd
import numpy as np
import pickle
import time
import os
from tqdm import tqdm
from datetime import timedelta
from operator import itemgetter
from geopy.distance import geodesic


def parse_raw(folders, params, length=1):
    # gather file names and path
    files = []
    for folder in folders:
        folder_files = filter(lambda f: f.startswith('v1'), os.listdir(folder))
        folder_files = map(lambda f: os.path.join(folder, f), folder_files)
        files.extend(list(folder_files))
    # gather files content
    raws, unsuccessful = [], []
    for file in tqdm(files):
        with open(file, 'rb') as f:
            try:
                raw = pickle.load(f)
                raw = list(filter(lambda r: True in [param in r[1] for param in params], raw))
                raws.extend(raw)
            except (MemoryError, EOFError, pickle.UnpicklingError):
                unsuccessful.append(file)
    # tabulate as dataframe with datetime index
    df = []
    for param in set(map(itemgetter(1), raws)):
        # epoch_param_value
        datum = np.array(list(filter(lambda epv: epv[1] == param, raws)))
        epoch = datum[:, 0]
        if length == 1:
            value = {param: datum[:, 2]}
        else:
            value = {param + f"_{i}": datum[:, 2+i] for i in range(length)}
        df.append(pd.DataFrame(value, index=epoch))
    df = pd.concat(df)
    df.index = pd.Series(df.index).apply(lambda ts: pd.Timestamp(float(ts), unit='s'))
    df = df.astype(float)
    if len(unsuccessful) > 0:
        print('files not parsed:')
        for f in unsuccessful:
            print(f)
    return df


def create_segments(unchopped, resampling='1S', gap=300, min_data_points=0.5):
    min_data_points *= gap
    gap = timedelta(seconds=gap)
    # find gap mask, False is gap larger than segment_gap
    mask = unchopped.index.to_list()
    mask = [True] + list(map(lambda z: z[1] - z[0] <= gap, zip(mask[:-1], mask[1:])))
    print('number of segments:', mask.count(False))
    segments, bgn = [], 0
    time.sleep(0.25)
    # fill segments
    for _ in tqdm(range(mask.count(False))):
        end = mask.index(False)
        segment = unchopped.iloc[bgn:end]
        if segment.index[-1] - segment.index[0] >= gap and segment.shape[0] >= min_data_points:
            # segment is larger than gap
            segment = segment.fillna(method='ffill').fillna(method='bfill')
            segment.index = segment.index.tz_localize('Asia/Singapore')
            segment = segment.resample('1S').ffill()
            segments.append(segment)
        mask[end] = True
        bgn = end
    time.sleep(0.25)
    print('accepted segments:', len(segments))
    return segments


def find_approx(sig):
    if sig.shape[0] == 1:
        return sig[0]
    else:
        return find_approx((sig[::2] + sig[1::2]) / 2)


def filter_signal(signal, window=3):
    window = 2**window
    sig = signal.to_numpy()
    threshold = [find_approx(sig[i:i+window]) for i in range(sig.shape[0]-window)]
    mask = (signal == 0) & (signal < np.array(threshold[:1] * window + threshold))
    signal = signal[~mask]
    return signal


def import_location(import_from):
    columns = ['mmsi', 'status', 'speed', 'lon', 'lat', 'course', 'heading', 'ts']
    # import
    location = [pd.read_csv(f) for f in import_from]
    location = pd.concat(location, ignore_index=True).reset_index(drop=True)
    location.columns = columns
    # datetime index
    location.index = pd.to_datetime(location['ts'], format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Asia/Singapore')
    # trim columns
    location = location.drop(columns=['mmsi', 'status', 'ts'])
    return location


def create_location_params(location):
    location['ts'] = location.index.to_series()
    # reset index
    location = location.reset_index(drop=True)
    # shift
    _lat, _lon = location['lat'].drop(location.shape[0]-1), location['lon'].drop(location.shape[0]-1)
    _idx = pd.to_datetime(location['ts'].drop(location.shape[0]-1), format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Asia/Singapore')
    location = location.drop(0).reset_index(drop=True)
    location['_lat'], location['_lon'] = _lat, _lon
    location['t0'] = _idx
    # displacement and direction (degrees ccw from north)
    latlon = location.loc[:, ('_lat', '_lon', 'lat', 'lon')].copy()
    location['displacement'] = latlon.apply(lambda x: geodesic(x[:2], x[2:]).meters, axis=1)
    location['direction'] = latlon.apply(lambda x: np.rad2deg(np.arctan((x[3]-x[1])/(x[2]-x[0])))%360, axis=1)
    # datetime index
    location.index = pd.to_datetime(location['ts'], format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Asia/Singapore')
    location['t1'] = location.index
    # location['timedelta'] = (location['t1']-location['t0']).astype(np.timedelta64(1, 's'))
    return location


def find_segments_location(segments, import_from, interval=1):
    location = import_location(import_from)
    location = location.resample('1S').interpolate(method='linear')
    locales = []
    for segment in tqdm(segments):
        mask = (location.index >= segment.index[0] - timedelta(seconds=interval)) & (location.index <= segment.index[-1])
        locale = location.loc[mask, ('lat', 'lon')]
        # if locale.shape[0] == 0:
        #     # no location info, set as last known location
        #     locale = location.loc[location.index <= segment.index[0], ('lat', 'lon')].iloc[-1:]
        #     locale.index = segment.index[:1]
        # if locale.index[0] != segment.index[0]:
        #     locale = pd.concat([pd.DataFrame(index=segment.iloc[:1].index), locale], sort=True)
        # if locale.index[-1] != segment.index[-1]:
        #     locale = pd.concat([locale, pd.DataFrame(index=segment.iloc[-1:].index)], sort=True)
        # locale = locale.resample('1T').mean().fillna(method='ffill').fillna(method='bfill')
        locale = create_location_params(locale)
        locales.append(locale)
    return locales
