import os
import pandas as pd
import numpy as np
import torch

def extract_features(dataset):
    """May need to extract features from the original 
       data. For now, try to put all raw data into nn and see how it performs

    Args:
        ...
    """
    pass

def create_dataset(features, targets, window_size=10, time_into_future=0.5, log_dt=0.02):
    """Transform a time series into a prediction dataset
    
    Args:
        features: A numpy array of feature sequences, shape=(num_timesteps, num_features)
        targets: A numpy array of target sequences, shape=(num_timesteps, 1)
        window_size: Size of window needed to be inputted to nn for prediction
        time_into_future: time into future (seconds) to predict
        log_dt: the logging time (s) between each state for each trajectory
    """
    assert len(features) == len(targets), "feature dimension should match that of target"
    num_timesteps_into_future = int(time_into_future / log_dt)
    X, y = [], []
    for i in range(len(features) - window_size - num_timesteps_into_future + 1):
        feature = features[i : i+window_size]
        target = targets[i + window_size + num_timesteps_into_future - 1]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

def process_csv_into_dataset(data_dir):
    """Transform all csv files under data_dir into a single concatenated 2D-ndarray
    
    Args:
        data_dir: directory where the csv files are located
    """
    Xs = None
    ys = None
    for csv_file in os.listdir(data_dir):
        traj = pd.read_csv(f"{data_dir}/{csv_file}").to_numpy()
        features = traj[:, :-1]
        targets = traj[:, -1]
        X, y = create_dataset(features=features, targets=targets)
        if Xs is not None:
            Xs = np.vstack((Xs, X))
        else:
            Xs = X
        if ys is not None:
            ys = np.hstack((ys, y))
        else:
            ys = y

    # Xs shape: (total_num_of_data, window_size, num_features)
    # ys shape: (total_num_of_data, )
    return Xs, ys 

