import os
import pandas as pd
import torch

def extract_features(dataset):
    """May need to extract features from the original 
       data. For now, try to put all raw data into nn and see how it performs

    Args:
        ...
    """
    pass

def create_dataset(features, targets, window_size, time_into_future=0.5, log_dt=0.02):
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
    return torch.tensor(X), torch.tensor(y)