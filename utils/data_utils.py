from abc import ABC, abstractmethod
import h5py
import random
import numpy as np
import json
import math
import quaternion
import os
import warnings
from os import path as osp
import sys
from joblib import Parallel, delayed

from utils.math_util import gyro_integration

"""
We use two levels of hierarchy for flexible data loading pipeline:
  - Sequence: Read the sequence from file and compute per-frame feature and target.
  - Dataset: subclasses of PyTorch's Dataset class. It has three roles:
      1. Create a Sequence instance internally to load data and compute feature/target.
      2. Apply post processing, e.g. smoothing or truncating, to the loaded sequence.
      3. Define how to extract samples from the sequence.


To define a new dataset for training/testing:
  1. Subclass CompiledSequence class. Load data and compute feature/target in "load()" function.
  2. Subclass the PyTorch Dataset. In the constructor, use the custom CompiledSequence class to load data. You can also
     apply additional processing to the raw sequence, e.g. smoothing or truncating. Define how to extract samples from 
     the sequences by overriding "__getitem()__" function.
  3. If the feature/target computation are expensive, consider using "load_cached_sequence" function.
  
Please refer to GlobalSpeedSequence and DenseSequenceDataset in data_global_speed.py for reference. 
"""


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """
    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass

    def get_meta(self):
        return "No info available"


def load_single_sequence(seq_type, root_dir, seq_name, cache_path, **kwargs):
    """
    Helper function to load a single sequence, used for parallel processing.
    """
    if cache_path is not None and osp.exists(osp.join(cache_path, seq_name + '.hdf5')):
        with h5py.File(osp.join(cache_path, seq_name + '.hdf5'), 'r') as f:
            feat = np.copy(f['feature'])
            targ = np.copy(f['target'])
            aux = np.copy(f['aux'])
            # No meta info printed when loading from cache to reduce spam in parallel mode
            return feat, targ, aux
    else:
        seq = seq_type(osp.join(root_dir, seq_name), **kwargs)
        feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
        print(seq.get_meta()) # Print meta info when loading raw data
        
        if cache_path is not None and osp.isdir(cache_path):
            try:
                with h5py.File(osp.join(cache_path, seq_name + '.hdf5'), 'x') as f:
                    f['feature'] = feat
                    f['target'] = targ
                    f['aux'] = aux
            except OSError as e:
                # Handle potential race conditions or file access issues gracefully
                print(f"Warning: Could not cache sequence {seq_name}: {e}")
        
        return feat, targ, aux


def load_cached_sequences(seq_type, root_dir, data_list, cache_path, **kwargs):
    grv_only = kwargs.get('grv_only', True)

    if cache_path is not None and cache_path not in ['none', 'invalid', 'None']:
        if not osp.isdir(cache_path):
            os.makedirs(cache_path)
        if osp.exists(osp.join(cache_path, 'config.json')):
            info = json.load(open(osp.join(cache_path, 'config.json')))
            if info['feature_dim'] != seq_type.feature_dim or info['target_dim'] != seq_type.target_dim:
                warnings.warn('The cached dataset has different feature or target dimension. Ignore')
                cache_path = 'invalid'
            if info.get('aux_dim', 0) != seq_type.aux_dim:
                warnings.warn('The cached dataset has different auxiliary dimension. Ignore')
                cache_path = 'invalid'
            if info.get('grv_only', 'False') != str(grv_only):
                warnings.warn('The cached dataset has different flag in "grv_only". Ignore')
                cache_path = 'invalid'
        else:
            info = {'feature_dim': seq_type.feature_dim, 'target_dim': seq_type.target_dim,
                    'aux_dim': seq_type.aux_dim, 'grv_only': str(grv_only)}
            json.dump(info, open(osp.join(cache_path, 'config.json'), 'w'))

    # Parallel data loading
    # Use n_jobs=-1 to use all available cores, or set a specific number like 16
    # Prefer 'loky' backend for stability with h5py, though 'threading' might be faster for I/O bound if GIL isn't an issue.
    # Given the heavy numpy and h5py usage, 'loky' (process-based) is generally safer but has higher overhead.
    # Let's use n_jobs=16 as requested/suggested for 64GB RAM servers, but capped by CPU count internally by joblib usually.
    # Set verbose=5 to see progress.
    
    print(f"Loading {len(data_list)} sequences with parallel processing...")
    results = Parallel(n_jobs=16, verbose=5, backend="loky")(
        delayed(load_single_sequence)(seq_type, root_dir, seq_name, cache_path, **kwargs)
        for seq_name in data_list
    )
    
    features_all, targets_all, aux_all = zip(*results)
    
    return list(features_all), list(targets_all), list(aux_all)


def select_orientation_source(data_path, max_ori_error=20.0, grv_only=True, use_ekf=True):

    """
    Select orientation from one of gyro integration, game rotation vector or EKF orientation.

    Args:
        data_path: path to the compiled data. It should contain "data.hdf5" and "info.json".
        max_ori_error: maximum allow alignment error.
        grv_only: When set to True, only game rotation vector will be used.
                  When set to False:
                     * If game rotation vector's alignment error is smaller than "max_ori_error", use it.
                     * Otherwise, the orientation will be whichever gives lowest alignment error.
                  To force using the best of all sources, set "grv_only" to False and "max_ori_error" to -1.
                  To force using game rotation vector, set "max_ori_error" to any number greater than 360.


    Returns:
        source_name: a string. One of 'gyro_integration', 'game_rv' and 'ekf'.
        ori: the selected orientation.
        ori_error: the end-alignment error of selected orientation.
    """
    ori_names = ['gyro_integration', 'game_rv']
    ori_sources = [None, None, None]

    with open(osp.join(data_path, 'info.json')) as f:
        info = json.load(f)
        ori_errors = np.array(
            [info['gyro_integration_error'], info['grv_ori_error'], info['ekf_ori_error']])
        init_gyro_bias = np.array(info['imu_init_gyro_bias'])

    with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
        ori_sources[1] = np.copy(f['synced/game_rv'])
        if grv_only or ori_errors[1] < max_ori_error:
            min_id = 1
        else:
            if use_ekf:
                ori_names.append('ekf')
                ori_sources[2] = np.copy(f['pose/ekf_ori'])
            min_id = np.argmin(ori_errors[:len(ori_names)])
            # Only do gyro integration when necessary.
            if min_id == 0:
                ts = f['synced/time']
                gyro = f['synced/gyro_uncalib'] - init_gyro_bias
                ori_sources[0] = gyro_integration(ts, gyro, ori_sources[1][0])

    return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]
