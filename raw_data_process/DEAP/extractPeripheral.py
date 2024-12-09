import pandas as pd
import numpy as np
import pyedflib  # 读取BDF数据
import matplotlib.pyplot as plt
import os
import math


# 信号处理
from scipy import signal    
from scipy.fftpack import fft, ifft, fftshift   # 傅里叶变换，逆傅里叶变换，将零频点移到频谱中间
from scipy.signal import welch, butter, filtfilt, find_peaks
import librosa
import hrvanalysis
import mne

class Pan_tompkins:
    """ Implementationof Pan Tompkins Algorithm.
    Noise cancellation (bandpass filter) -> Derivative step -> Squaring and integration.
    Params:
        data (array) : ECG data
        sampling rate (int)
    returns:
        Integrated signal (array) : This signal can be used to detect peaks
    ----------------------------------------
    HOW TO USE ?
    Eg.
    ECG_data = [4, 7, 80, 78, 9], sampling  =2000
    
    call : 
       signal = Pan_tompkins(ECG_data, sampling).fit()
    ----------------------------------------
    
    """

    def __init__(self, data, sample_rate):

        self.data = data
        self.sample_rate = sample_rate

    def fit(self, normalized_cut_offs=None, butter_filter_order=2, padlen=150, window_size=None):
        ''' Fit the signal according to algorithm and returns integrated signal
        
        '''
        # 1.Noise cancellationusing bandpass filter
        self.filtered_BandPass = self.band_pass_filter(
            normalized_cut_offs, butter_filter_order, padlen)

        # 2.derivate filter to get slpor of the QRS
        self.derviate_pass = self.derivative_filter()

        # 3.Squaring to enhance dominant peaks in QRS
        self.square_pass = self.squaring()

        # 4.To get info about QRS complex
        self.integrated_signal = self.moving_window_integration(window_size)

        return self.integrated_signal

    def band_pass_filter(self, normalized_cut_offs=None, butter_filter_order=2, padlen=150):
        ''' Band pass filter for Pan tompkins algorithm
            with a bandpass setting of 5 to 20 Hz
            params:
                normalized_cut_offs (list) : bandpass setting canbe changed here
                bandpass filte rorder (int) : deffault 2
                padlen (int) : padding length for data , default = 150
                        scipy default value = 2 * max(len(a coeff, b coeff))
            return:
                filtered_BandPass (array)
        '''

        # Calculate nyquist sample rate and cutoffs
        nyquist_sample_rate = self.sample_rate / 2

        # calculate cutoffs
        if normalized_cut_offs is None:
            normalized_cut_offs = [
                5/nyquist_sample_rate, 15/nyquist_sample_rate]
        else:
            assert type(
                self.sample_rate) is list, "Cutoffs should be a list with [low, high] values"

        # butter coefficinets
        b_coeff, a_coeff = butter(
            butter_filter_order, normalized_cut_offs, btype='bandpass')[:2]

        # apply forward and backward filter
        filtered_BandPass = filtfilt(
            b_coeff, a_coeff, self.data, padlen=padlen)

        return filtered_BandPass

    def derivative_filter(self):
        ''' Derivative filter
        params:
            filtered_BandPass (array) : outputof bandpass filter
        return:
            derivative_pass (array)
        '''

        # apply differentiation
        derviate_pass = np.diff(self.band_pass_filter())

        return derviate_pass

    def squaring(self):
        ''' squaring application on derivate filter output data
        params:
        return:
            square_pass (array)
        '''

        # apply squaring
        square_pass = self.derivative_filter() ** 2

        return square_pass

    def moving_window_integration(self, window_size=None):
        ''' Moving avergae filter 
        Params:
            window_size (int) : no. of samples to average, if not provided : 0.08 * sample rate
            sample_rate (int) : should be given if window_size is not given  
        return:
            integrated_signal (array)
        '''

        if window_size is None:
            assert self.sample_rate is not None, "if window size is None, sampling rate should be given"
            # given in paper 150ms as a window size
            window_size = int(0.08 * int(self.sample_rate))

        # define integrated signal
        integrated_signal = np.zeros_like(self.squaring())

        # cumulative sum of signal
        cumulative_sum = self.squaring().cumsum()

        # estimationof area/ integral below the curve deifnes the data
        integrated_signal[window_size:] = (
            cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

        integrated_signal[:window_size] = cumulative_sum[:window_size] / \
            np.arange(1, window_size + 1)

        return integrated_signal

def butter_lowpass(cutoff, fs, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def extract_GSR_features(gsr_trial, fs=256):
    """
    提取GSR信号的各类特征
    
    参数:
    gsr_trial: List of trials, 每个trial是一个numpy数组，表示一段GSR数据
    fs: 采样频率，默认为256Hz
    
    返回:
    特征字典
    """
    
    # 初始化特征字典
    features = {
        "ASR": [],        # 平均皮肤电阻
        "ABD": [],        # 平均绝对导数
        "PNVA": [],       # 负导数比例
        "PSD": [],        # 功率谱密度
        "SCSR_ZCR": [],   # SCSR的零交叉率
        "SCVSR_ZCR": [],  # SCVSR的零交叉率
        "SCSR_MPM": [],   # SCSR的峰值幅度均值
        "SCVSR_MPM": []   # SCVSR的峰值幅度均值
    }
    
    for trial in gsr_trial:
        # 提取GSR信号（假设GSR数据在每个trial的第4列）
        gsr_signal = trial[:, 3]
        
        # ASR: 平均皮肤电阻
        features["ASR"].append(np.mean(gsr_signal))
        
        # ABD: 平均绝对导数
        smoothed_signal = filtfilt(*butter_lowpass(0.5, fs), gsr_signal)  # 平滑信号
        derivative = np.diff(smoothed_signal)  # 计算导数
        features["ABD"].append(np.mean(np.abs(derivative)))  # 计算平均绝对导数
        
        # PNVA: 负导数比例
        features["PNVA"].append(np.mean(derivative < 0))  # 计算负导数比例
        
        # PSD: 功率谱密度
        b, a = butter_lowpass(2.4, fs)
        filtered_signal = filtfilt(b, a, gsr_signal)  # 低通滤波
        f, Pxx = welch(filtered_signal, fs=fs, nperseg=1280)  # 计算功率谱密度
        features["PSD"].append(Pxx[:13])  # 取[0, 2.4]Hz范围内的前13个频率分量
        
        # SCSR ZCR: Skin Conductance Slow Response的零交叉率
        b, a = butter_lowpass(0.2, fs)
        SCSR = filtfilt(b, a, gsr_signal)  # 滤波获取慢反应部分
        features["SCSR_ZCR"].append(np.sum(librosa.zero_crossings(SCSR)))  # 计算零交叉率
        
        # SCVSR ZCR: Skin Conductance Very Slow Response的零交叉率
        b, a = butter_lowpass(0.08, fs)
        SCVSR = filtfilt(b, a, gsr_signal)  # 滤波获取非常慢反应部分
        features["SCVSR_ZCR"].append(np.sum(librosa.zero_crossings(SCVSR)))  # 计算零交叉率
        
        # SCSR MPM: SCSR的峰值幅度均值
        peaks_SCSR, _ = find_peaks(SCSR)
        if len(peaks_SCSR) > 0:
            features["SCSR_MPM"].append(np.mean(SCSR[peaks_SCSR]))  # 计算SCSR的峰值幅度均值
        else:
            features["SCSR_MPM"].append(0)  # 如果没有峰值，返回0
        
        # SCVSR MPM: SCVSR的峰值幅度均值
        peaks_SCVSR, _ = find_peaks(SCVSR)
        if len(peaks_SCVSR) > 0:
            features["SCVSR_MPM"].append(np.mean(SCVSR[peaks_SCVSR]))  # 计算SCVSR的峰值幅度均值
        else:
            features["SCVSR_MPM"].append(0)  # 如果没有峰值，返回0
    
    return features

def extractECG(peripheral_trial_list):
    ECG = peripheral_trial_list
    # for i in range(len(peripheral_trial_list)):
        # ECG.append(peripheral_trial_list[i][:,2])
    HRV = []    # 变异系数
    RMSSD = []  # root mean square of the mean squared difference of successive beats
    PSD = []  # 56 spectral power in the bands from [0,6]Hz
    LFPS = []   # low frequency [0.01, 0.08]Hz components of HRV power spectrum
    MFPS = []   # medium frequency [0.08, 0.15]Hz components of HRV power spectrum
    HFPS = []   # high frequency [0.15, 0.5]Hz components of HRV power spectrum
    SD1 = []    # poincaré analysis features 1
    SD2 = []    # poincaré analysis features 2
    ECG = np.asarray(ECG)
    for trial in ECG:
        QRS = Pan_tompkins(trial,sample_rate=256).fit() #   检测QRS波群
        
        R_peaks_index = find_peaks(QRS,distance=175)[0]
        R_peaks_mean = QRS[R_peaks_index].mean()
        # 剔除可能不属于峰值的值
        remove_index = []
        for index in range(0,len(R_peaks_index)):
            if QRS[R_peaks_index[index]]<R_peaks_mean/2:
                remove_index.append(index)
        R_peaks_index = np.delete(R_peaks_index,remove_index)
        RR_intervals = R_peaks_index[1:]-R_peaks_index[0:-1]

        time_domain_features = hrvanalysis.get_time_domain_features(RR_intervals)
        HRV.append(RR_intervals.mean())     # 怀疑其实是平均值

        RMSSD.append(time_domain_features['rmssd']) 

        b,a = butter(1,6,"lowpass",fs=256)
        filterData = filtfilt(b,a,trial)
        freq,Pxx = welch(filterData,fs=256,nperseg=128*55/3)
        PSD.append(Pxx[:56])

        b,a = butter(1,[0.01,0.08],"bandpass",fs=256)
        filterData = filtfilt(b, a, RR_intervals)
        freq,Pxx = welch(filterData,fs=256)
        LFPS.append(np.sum(Pxx))

        b,a = butter(1,[0.08,0.15],"bandpass",fs=256)
        filterData = filtfilt(b, a, RR_intervals)
        freq,Pxx = welch(filterData,fs=256)
        MFPS.append(np.sum(Pxx))

        b, a = butter(1, [0.15, 0.5], "bandpass", fs=256)
        filterData = filtfilt(b, a, RR_intervals)
        freq, Pxx = welch(filterData, fs=256)
        HFPS.append(np.sum(Pxx))

        poincare_plot_features = hrvanalysis.get_poincare_plot_features(RR_intervals)
        SD1.append(poincare_plot_features['sd1'])
        SD2.append(poincare_plot_features['sd2'])

        # print("RR_intervals:",RR_intervals)
        # print("HRV:",HRV)
        # print("RMSSD:",RMSSD)
        # print("PSD:",PSD)
        # print("LFPS:",LFPS)
        # print("MFPS:",MFPS)
        # print("HFPS:",HFPS)
        # print("SD1:",SD1)
        # print("SD2:",SD2)
    return HRV,RMSSD,PSD,LFPS,MFPS,HFPS,SD1,SD2

# Respiration pattern（呼吸方式)
def extractRespirationFeature(peripheral_trial_list):
    # RSP = []
    # for i in range(len(peripheral_trial_list)):
        # RSP.append(peripheral_trial_list[i][:,4])
    RSP = np.array(peripheral_trial_list)
    RANGE = []  # range
    # band energy ratio(difference between the logarithm of energy between the lower([0.05,0.25]Hz)) and the higher([0.25,5]Hz bands
    BEN = []
    VRS = []    # mean of derivative （variation of the respiration signal)
    BRSC = []   # breathing rhythm(spectral centroid)
    BRV = []    # average breathe depth(peak to peak)
    PSD = []    # 8 spectral power in the bands in [0, 2.4]Hz
    for trial in RSP:
        b,a = butter(1,[0.05,0.25],'bandpass',fs=256)
        filterData = filtfilt(b,a,trial)
        freq,Pxx = welch(filterData,fs=256)
        low_freq = np.average(Pxx)

        b,a = butter(1,[0.25,5],'bandpass',fs=256)
        filterData = filtfilt(b,a,trial)
        freq,Pxx = welch(filterData,fs=256)
        high_freq = np.average(Pxx)
        BEN.append(np.log(low_freq)-np.log(high_freq))
        
        RANGE.append(trial.max()-trial.min())   # Dynamic range

        VRS.append(np.mean(trial[1:]-trial[:-1]))
        BRSC.append(librosa.feature.spectral_centroid(trial,sr=256)[0])

        index = find_peaks(trial,width=70)[0]
        BRV.append(np.average(index[1:]-index[:-1]))

        b, a = butter(1, 2.4, 'lowpass', fs=256)
        filterData = filtfilt(b, a, trial)
        freq, Pxx = welch(filterData, fs=256,nperseg=746)
        PSD.append(Pxx[:8])

        # print("BEN:",BEN)
        # print("RANGE:",RANGE)
        # print("VRS:",VRS)
        # print("BRSC:",BRSC)
        # print("BRV:",BRV)
        # print("PSD:",PSD)
        return BEN,RANGE,VRS,BRSC,BRV,PSD

def extractSkinTemperatureFeature(peripheral_trial_list):
    ST = []
    for i in range(len(peripheral_trial_list)):
        ST.append(peripheral_trial_list[i][:,5])
    AVG = []    # average.
    AD = []     # average of its derivative
    PSD1 = []   # spectral power in the bands（[0-0.1]Hz，[0.1 - 0.2]Hz
    PSD2 = []
    for trial in ST:
        AVG.append(trial.mean())
        AD.append(np.mean(trial[1:]-trial[:-1]))

        b,a = butter(1,0.1,"lowpass",fs=256)
        filterData = filtfilt(b,a,trial)
        f,Pxx = welch(filterData,fs=256)
        PSD1.append(Pxx.mean())

        b,a = butter(1,[0.1,0.2],"bandpass",fs=256)
        filterData = filtfilt(b,a,trial)
        f,Pxx = welch(filterData,fs=256)
        PSD2.append(Pxx.mean())
        
        # print("AVG：",AVG)
        # print("AD：",AD)
        # print("PSD1:",PSD1)
        # print("PSD2:",PSD2)
    return AVG,AD,PSD1,PSD2