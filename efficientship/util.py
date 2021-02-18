import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import swifter
import pickle
import time
import os
from tqdm import tqdm
from itertools import chain
from datetime import timedelta
from operator import itemgetter
from sklearn.metrics import auc
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from scipy.ndimage import gaussian_filter
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed


# chunker function to chunk signal into subsets
chunker = lambda signal, i, stride, time_constant: np.array(signal[i*stride:i*stride+time_constant]).reshape((-1, 1))


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
    location['distance'] = latlon.apply(lambda x: geodesic(x[:2], x[2:]).meters, axis=1)
    location['direction'] = latlon.apply(lambda x: np.rad2deg(np.arctan((x[3]-x[1])/(x[2]-x[0])))%360, axis=1)
    # datetime index
    location.index = pd.to_datetime(location['ts'], format='%Y-%m-%d %H:%M:%S', utc=True).dt.tz_convert('Asia/Singapore')
    location['t1'] = location.index
    # location['timedelta'] = (location['t1']-location['t0']).astype(np.timedelta64(1, 's'))
    return location


def find_segments_location(segments, import_from, interval=1):
    location = pd.concat([pd.read_csv(f) for f in import_from], ignore_index=True)
    location.columns = ['mmsi', 'status', 'speed', 'lon', 'lat', 'course', 'heading', 'ts']
    location.index = pd.to_datetime(location['ts'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize('Asia/Singapore')
    location = location.loc[:, ('lon', 'lat')]
    location = location.resample('1S').interpolate(method='linear')
    location['_lon'], location['_lat'] = location['lon'].shift(periods=-1), location['lat'].shift(periods=-1)
    location = location.drop(location.index[-1])
    locales = []
    for segment in tqdm(segments):
        locale = location.loc[(location.index >= segment.index[0]) & (location.index <= segment.index[-1]), ('_lat', '_lon', 'lat', 'lon')]
        locale['distance'] = locale.apply(lambda row: geodesic(row[:2], row[-2:]).meters, axis=1)
        # locale = create_location_params(locale)
        locales.append(locale)
    return locales


def ts_clustering(X_train, stride, time_constant, n_cluster):
    # split each segment into chunks, then flatten
    X_train = [chunker(signal, i, stride, time_constant) for signal in X_train for i in range((len(signal)-time_constant)//stride)]
    # transform to tslearn dataset
    X_train = to_time_series_dataset(X_train)
    # clustering and train
    km = TimeSeriesKMeans(n_clusters=n_cluster, metric='euclidean')
    km.fit(X_train)
    # hasher, to turn label to score
    means = [(i, np.mean(agg)) for i, agg in enumerate(km.cluster_centers_)]
    hasher = dict(zip(map(itemgetter(0), sorted(means, key=itemgetter(1))), range(len(means))))
    return km, hasher


def get_scores(segments, segments_wind, segments_location, stride=15, time_constant=60, n_cluster=10):
    # prepare signals
    signals_mflow = [segment.mean(axis=1).to_numpy() for segment in segments]
    signals_wind = [segment['effect'].to_numpy() for segment in segments_wind]
    signals_loc = [segment['distance'].to_numpy() for segment in segments_location]
    signals = [signals_mflow, signals_wind, signals_loc]
    # create and train models, create hashes
    estimators = [ts_clustering(segments, stride, time_constant, n_cluster) for segments in signals]
    # convert to scores
    scores = []
    for (estimator, hash), signal in zip(estimators, signals):
        score = []
        for segment in signal:
            chunk = [chunker(segment, i, stride, time_constant) for i in range((len(segment)-time_constant)//stride)]
            labels = estimator.predict(chunk)
            score.append(np.array([hash[label] for label in labels]))
        scores.append(score)
    return signals, scores, estimators

def generate_initial_proba(segments, n_class, cluster_name):
    # segments = list(chain.from_iterable(segments))
    segments = [segment[0] for segment in segments]
    unique, counts = np.unique(segments, return_counts=True)
    proba = dict(zip(range(n_class), np.zeros((n_class, ))))
    proba = {**dict(zip(range(n_class), np.zeros((n_class, )))), **dict(zip(unique, counts/len(segments)))}
    # for k, v in dict(zip(unique, counts/len(segments))).items():
    #     proba[k] = v
    proba = map(itemgetter(1), sorted([(k, v) for k, v in proba.items()], key=itemgetter(0)))
    return cluster_name, np.array(list(proba))

def generate_hidden_matrix(segments, n_class, cluster_name):
    seq_pairs, states = [], []
    for segment in segments:
        states.extend(list(segment))
        seq_pairs.extend([segment[i:i+2] for i in range(len(segment)-1)])
    states = set(states)
    # distribution
    unique, counts = np.unique(list(map(lambda pair: pair[1] - pair[0], seq_pairs)), return_counts=True)
    counts = counts / np.sum(counts)
    distribution = dict(zip(unique, counts))
    # find next states
    next_states = []
    for group in map(lambda hidden_state: list(filter(lambda pair: pair[0] == hidden_state, seq_pairs)), states):
        this_state = set(map(itemgetter(0), group))
        if len(this_state) != 1: 
            continue
        else: 
            this_state = list(this_state)[0]
        next_state = list(map(itemgetter(1), group))
        unique, counts = np.unique(next_state, return_counts=True)
        proba = counts / len(next_state)
        next_states.append((this_state, dict(zip(unique, proba))))
    next_states = dict(next_states)
    # build dataframe, must account for missing states
    states = []
    for this_state in range(n_class):
        # all zero
        next_state = dict(zip(range(n_class), np.zeros((n_class, ))))
        if this_state in next_states:
            # overwrite
            next_state = {**next_state, **next_states[this_state]}
        else:
            next_state = {**next_state, **{(k + this_state): v for k, v in distribution.items() if 0 <= (k + this_state) <= n_class}}
        states.append(pd.DataFrame(next_state, index=[this_state]))
    states = pd.concat(states, sort=True)
    return cluster_name, states.to_numpy()

def generate_observable_matrix(segments_hidden, segments_observable, n_class, cluster_name):
    association = list(zip([s for segment in segments_hidden for s in segment], [s for segment in segments_observable for s in segment]))
    association_df = []
    for this_state in range(n_class):
        class_association = dict(zip(range(n_class), np.zeros((n_class, ))))
        class_members = list(filter(lambda z: z[0] == this_state, association))
        if len(class_members) != 0:
            evident_class_association = list(map(itemgetter(1), class_members))
            unique, counts = np.unique(evident_class_association, return_counts=True)
            class_association = {**class_association, **dict(zip(unique, counts/len(evident_class_association)))}
        else:
            # print('no association:', this_state)
            pass
        association_df.append(pd.DataFrame(class_association, index=[this_state]))
    association_df = pd.concat(association_df, sort=True)
    return cluster_name, association_df.to_numpy()

# find datapoints cluster member
def associate_to_cluster(data, centers):
    distance_to_centers = np.array([np.linalg.norm(data - center, axis=1) for center in centers]).T
    minimum_distance = np.array([np.min(distance) for distance in distance_to_centers])
    cluster_member_mask = np.array([distances == minimum for distances, minimum in zip(distance_to_centers, minimum_distance)])
    cluster_member_card = np.array([np.ones((data.shape[0], )) * i for i in range(len(centers))]).T
    datapoint_labels = np.array([card[mask][0] for card, mask in zip(cluster_member_card, cluster_member_mask)])
    return datapoint_labels
    
def get_cluster_member(score, cluster_centers):
    seg = np.array(score).T
    dist = np.array([np.linalg.norm(seg - center, axis=1) for center in cluster_centers]).T
    minim = np.min(dist, axis=1)
    mask = np.array([d == minim for d in dist.T]).T
    cluster_member = np.array([np.ones((mask.shape[0], )) * i for i in range(np.shape(cluster_centers)[0])]).T[mask]
    return cluster_member

def viterbi(pi, a, b, obs):
    
    n_states = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = path = np.zeros(T, dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((n_states, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((n_states, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    # print('\nstart walking forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(n_states):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            # print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    # print('-' * 50)
    # print('start backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        # print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi


def test_all_segments(scores, n_class, n_cluster=None, is_plot=False):
    """
    docstring
    """
    n_segment = len(scores[0])
    print('number of segments:', n_segment, '\n')
    paths = []
    for test_segment in range(n_segment):

        t0 = time.time()
        # bias is for stretching/squeezing a particular axial space 
        # to purposefully make points closer or away
        # lower is closer
        biases = [1, 1, 1]
        # epsilon is minimum distance of outermost cluster point to the next unclustered point 
        # to absorb as cluster member
        epsilon = np.sqrt(8)

        # create datapoints, and apply bias
        b = np.array([score for enum, score in enumerate(scores[0]) if enum != test_segment])
        b = np.concatenate(b * biases[0])
        w = np.array([score for enum, score in enumerate(scores[1]) if enum != test_segment])
        w = np.concatenate(w * biases[1])
        d = np.array([score for enum, score in enumerate(scores[2]) if enum != test_segment])
        d = np.concatenate(d * biases[2])
        datapoints = np.array([b, w, d]).T

        # find good number of cluster
        if n_cluster is None:
            n_cluster = 5
            is_static_n_cluster = False
        else:
            is_static_n_cluster = True
        infinite_breaker = 1000
        cluster_quality_thresh = 1
        # k-means
        n_cluster_ = n_cluster
        cluster_centers = []
        while True:
            if n_cluster_ > infinite_breaker:
                # print('optimum not found')
                break
            # create clusters, without noise points
            est = KMeans(n_clusters=n_cluster_, n_jobs=-1)
            est.fit(datapoints)
            # inverse mean square error
            dfunc = lambda i: np.linalg.norm(datapoints[est.labels_==i] - est.cluster_centers_[i], axis=1)
            cluster_quality = np.array([dfunc(j) for j in range(n_cluster_)])
            cluster_quality = n_class / np.array([np.mean(cq**2) for cq in cluster_quality])
            # cluster labels accepted
            accepted = cluster_quality >= cluster_quality_thresh
            if not np.all(accepted) and not is_static_n_cluster:
                # print(f"n_cluster: {n_cluster_}, minimum: {np.min(cluster_quality)}")
                n_cluster_ += 5
            else:
                break
        # print('\nnumber of clusters:', est.cluster_centers_.shape[0], '\n')
        n_cluster = est.cluster_centers_.shape[0]

        # prepare for training
        dataset = {i: [] for i in range(n_cluster)}
        bgn = 0
        for enum, score in enumerate(zip(*scores)):
            if enum == test_segment:
                continue
            # segment points
            seg = np.array(score).T
            # find end then slice
            end = bgn + seg.shape[0]
            cluster_member = est.labels_[bgn:end]
            # for every cluster
            for i in range(n_cluster):
                # get index 
                sequence = np.where(cluster_member == i)[0]
                # must be 2 or more
                if sequence.shape[0] < 2:
                    continue
                ptr = sequence[0]
                for idx in sequence[1:]:
                    if idx - ptr != 1:
                        # not adjacent
                        i_member = seg[ptr:idx]
                        if i_member.shape[0] > 1:
                            # len is 2 or more
                            dataset[i].append(i_member)
                        ptr = idx
            bgn = end
        for k, v in dataset.items():
            if len(v) == 0:
                print(k, 'has no member')

        # create matrices
        # 0 bunker, 1 wind effect, 2 distance
        hidden_param = 0
        observable_param = 2
        # hidden: to be predicted
        # observable: input
        hmm = {}
        pi, hidden, observable = [], [], []
        with ProcessPoolExecutor() as executor:
            wait_for_1, wait_for_2, wait_for_3 = [], [], []
            for i in range(n_cluster):
                wait_for_1.append(executor.submit(generate_initial_proba, [seg[:, hidden_param] for seg in dataset[i]], n_class, i))
                wait_for_2.append(executor.submit(generate_hidden_matrix, [seg[:, hidden_param] for seg in dataset[i]], n_class, i))
                wait_for_3.append(executor.submit(generate_observable_matrix, [seg[:, hidden_param] for seg in dataset[i]], 
                                                                              [seg[:, observable_param] for seg in dataset[i]], 
                                                                              n_class, i))
            for f in as_completed(wait_for_1):
                pi.append(f.result())
            for f in as_completed(wait_for_2):
                hidden.append(f.result())
            for f in as_completed(wait_for_3):
                observable.append(f.result())
        pi = list(map(itemgetter(1), sorted(pi, key=itemgetter(0))))
        hidden = list(map(itemgetter(1), sorted(hidden, key=itemgetter(0))))
        observable = list(map(itemgetter(1), sorted(observable, key=itemgetter(0))))
        for i, zipped in enumerate(zip(pi, hidden, observable)):
            hmm[i] = zipped

        # get prediction path
        # some var names are recycled
        score = np.array([scores[i][test_segment] for i in range(3)])
        # cluster_member = get_cluster_member(score, est.cluster_centers_)
        cluster_member = est.predict(score.T)
        level = cluster_member[0]
        cluster_member_segments, one_level = [], [level]
        for i in range(1, cluster_member.shape[0]):
            if cluster_member[i] == level:
                one_level.append(cluster_member[i])
            else:
                cluster_member_segments.append(one_level)
                level = cluster_member[i]
                one_level = [level]
        if len(one_level) > 0:
            cluster_member_segments.append(one_level)
        path, bgn = [], 0
        for one_level in cluster_member_segments:
            end = bgn + len(one_level)
            # pi, hidden, observable = (gaussian_filter(p, sigma=0.5) for p in hmm[one_level[0]])
            pi, hidden, observable = hmm[one_level[0]]
            observable = gaussian_filter(observable, sigma=0.5)
            p, *_ = viterbi(pi, hidden, observable, scores[2][test_segment][bgn:end])
            path.extend(p)
            bgn = end  

        # plot
        if is_plot:
            plt.ylabel('Cluster')
            # plt.plot(cluster_member, color='k', linewidth=0.5)
            plt.bar(range(len(cluster_member)), cluster_member, width=1, facecolor='grey', alpha=0.25)
            plt.twinx()
            plt.ylabel('Score')
            plt.plot(score[hidden_param], label='FR');
            plt.plot(score[observable_param], label='VS');
            plt.xlabel('Sample')
            plt.plot(path, label='PR');
            plt.legend(loc='upper right');
            plt.show();

        # accuracy
        auc_truth = auc(range(len(score[hidden_param])), score[hidden_param])
        auc_pred = auc(range(len(path)), path)
        auc_diff = (auc_pred - auc_truth) / auc_truth
        print('# {:3.0f} | {:3.0f} seconds | {} clusters | off by: {:6.1f} %'.format(test_segment, 
                                                                                     round(time.time()-t0), 
                                                                                     n_cluster, 
                                                                                     100 * auc_diff))
        print()
        paths.append((score[hidden_param], path))
    return paths
