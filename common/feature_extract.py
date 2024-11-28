import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import scipy.sparse as sp
import os
import mne
from typing import List, Tuple
from mne.preprocessing import ICA
import scipy.signal as signal
from scipy.interpolate import griddata
from sklearn.preprocessing import normalize, scale
from mne.decoding import UnsupervisedSpatialFilter
from common.data_process import butterworth_filter

from sklearn.decomposition import PCA, FastICA

from sklearn import preprocessing

import math

import random
import time


#######################################################
####################### 时域特征 #######################

def get_engery(trial_data):
    """
    Introduction:
        一段trial的能量特征
    Args:
        trial_data: samples * channels
        return: trial engery feature
    """
    trial_data = trial_data.T
    channels = trial_data.shape[0]
    en=np.zeros(channels)
    p = np.power(trial_data, 2)
    for i in range(channels):
        en[i] = np.sum(p[i])
    return en.T

def __get_hjorth_activity(trial_data):
    """
    Introduction:
        一段trial的activity特征
    Args:
        trial_data: samples * channels
        return: trial activity feature
    """
    trial_data = trial_data.T
    channels = trial_data.shape[0]
    avg_s = np.average(trial_data,axis=1)
    for i in np.arange(channels):
        trial_data[i] = trial_data[i] - avg_s[i]
    p = np.power(trial_data,2)
    activity = np.average(p,axis=1)
    return activity.T

def __hjorth_mobility_complexity(channel_data):
    """
    Introduction:
        一个channel的mobility, complexity特征
    Args:
        channel_data: samples * 1
        return: trial mobility feature, trial complexity feature
    """
    D = np.diff(channel_data,axis=0)
    D = np.insert(D,0,0,axis=0)
    
    N = len(channel_data)
    
    M2 = np.sum(D ** 2, axis = 0) / N
    TP = np.sum(channel_data ** 2, axis = 0)
    M4 = 0
    for i in range(N-1):
        M4 += (D[i+1] - D[i]) ** 2
    M4 = M4 / N
    
    mobility = np.sqrt(M2 / TP)
    complexity = np.sqrt(M4 * TP / M2 / M2)
    
    return mobility, complexity

def __get_hjorth_mobility_complexity(trial_data):
    """
    Introduction:
        一个trial的mobility, complexity特征
    Args:
        trial_data: samples * channels
        return: trial mobility feature, trial complexity feature
    """
    trial_data = trial_data.T
    channels = trial_data.shape[0]

    mobility = np.zeros(channels)
    complexity = np.zeros(channels)

    for i in range(channels):
        mobility[i],complexity[i] = __hjorth_mobility_complexity(trial_data[i])

    return mobility.T, complexity.T

def get_hjorth(trial_data):
    """
    Introduction:
        一个trial的hjorth特征, 包括activity, mobility, complexity特征
    Args:
        trial_data: samples * channels
        return: trial hjorth
    """
    activity = __get_hjorth_activity(trial_data)
    mobility, complexity = __get_hjorth_mobility_complexity(trial_data)
    return np.concatenate([activity, mobility, complexity])


def get_all_timedomain_feature(trial_data):
    """
    Introduction:
        一个trial的所有时域特征
    Args:
        trial_data: samples * channels
        return: trial mobility feature, trial complexity feature
    """
    f1 = get_engery(trial_data)
    f2 = __get_hjorth_activity(trial_data)
    f3, f4 = __get_hjorth_mobility_complexity(trial_data)
    return np.concatenate([f1, f2, f3, f4])


#######################################################
####################### 频域特征 #######################

def compute_DE(trial_data, fs=256, band = [1,4,8,13,31,70]):
    """
    Introduction:
        一个trial的DE特征
    Args:
        trial_data: samples * channels
        fs: frequency of sample. Default to 256
        band: frequcency band of eeg signal. Default to [1,4,8,13,31,75]
        return: trial DE feature, shape bands * channels
    """
    trial_data = trial_data.T
    channels = trial_data.shape[0]
    de = np.zeros((channels, len(band)-1))
    fre_step = band
    for i in range(5):
        sub_fre = butterworth_filter(trial_data, fs, fre_step[i], fre_step[i+1], order=3)
        var = np.var(sub_fre, axis=1, ddof=1)
        raw_de = [(math.log(2 * math.pi * math.e * x) / 2) for x in var]
        raw_de = np.array(raw_de)
        # raw_de = (raw_de - raw_de.min())/(raw_de.max() - raw_de.min())
        de[:,i] = raw_de
    de = np.array(de.T)
    return de

def compute_power_spectral_density(trial_data, fs=256, band = [1,4,8,13,31,75], sliding_window = 500, overlap = 0.25):
    """
    Introduction:
        一个trial的PSD特征
    Args:
        trial_data: samples * channels
        fs: frequency of sample. Default to 256
        band: frequcency band of eeg signal. Default to [1,4,8,13,31,75]
        sliding_window: sliding_window. Default to 2*fs
        overlap: overlap. Default 0.25
        return: trial PSD feature, shape bands * channels
    """
    trial_data = trial_data.T
    ret = []
    fre_step = band
    n_overlap = int(sliding_window * overlap)
    # compute psd using Welch method
    freqs, power = signal.welch(trial_data, fs=fs, nperseg=sliding_window, noverlap=n_overlap)
    for i in range(5):
        tmp = (freqs >= fre_step[i]) & (freqs < fre_step[i + 1])
        ret.append(power[:,tmp].mean(1))

    return(np.log(np.array(ret) / np.sum(ret, axis=0)))

def __bin_power(channel_data, fs = 256, band = [1,4,8,13,31,75]):
    """
    Introduction:
        一个channel的bin_power特征
    Args:
        channel_data: samples * 1
        fs: frequency of sample. Default to 256
        band: frequcency band of eeg signal. Default to [1,4,8,13,31,75]
        return: trial PSD feature, shape bands * channels
    """
    # single channel
    C = np.fft.fft(channel_data)
    C = abs(C)
    Power = np.zeros(len(band) - 1)
    for Freq_Index in range(0, len(band) - 1):
        Freq = float(band[Freq_Index])
        Next_Freq = float(band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / fs * len(channel_data))): 
                int(np.floor(Next_Freq / fs * len(channel_data)))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio

def compute_bin_power(trial_data, fs=256, band = [1,4,8,13,31,75]):
    """
    Introduction:
        一个trial的bin_power特征
    Args:
        trial_data: samples * channels
        fs: frequency of sample. Default to 256
        band: frequcency band of eeg signal. Default to [1,4,8,13,31,75]
        return: trial bin_power feature, shape bands * channels
    """
    trial_data = trial_data.T
    channels = trial_data.shape[0]
    out, _ = __bin_power(trial_data[0], fs, band)
    for i in range(1, channels):
        tp, _ = __bin_power(trial_data[i], fs, band)
        out = np.vstack((out, tp))
    return out.T

def compute_all_frequency_feature(trial_data, fs = 256, band = [1,4,8,13,31,75]):
    """
    Introduction:
        一个trial的所有频域特征
    Args:
        trial_data: samples * channels
        fs: frequency of sample. Default to 256
        band: frequcency band of eeg signal. Default to [1,4,8,13,31,75]
        return: trial all frequency features
    """
    f1 = compute_power_spectral_density(trial_data, fs, band)
    f2 = compute_DE(trial_data, fs, band)
    f3 = compute_bin_power(trial_data, fs, band)
    return np.concatenate([f1, f2, f3], axis=1)