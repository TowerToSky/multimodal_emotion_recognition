import os
import mne
import numpy as np
import pandas as pd
from xml.dom.minidom import parse
from sklearn.decomposition import FastICA


def _get_eeg_pps_trial(eeg_pps_path):
    """
    Introduction:
        根据eeg_pps_path获取受试者trial
    Args:
        eeg_pps_path: 脑电和生理信号数据路径
        return: eeg trial, pps trail. trial shape: shample * channel
    """

    raw = mne.io.read_raw_bdf(eeg_pps_path, preload=True, verbose=False)
    # Exclude useless channels
    exclude_channels = ["EXG4", "EXG5", "EXG6", "EXG7", "EXG8", "GSR2", "Erg1", "Erg2"]
    raw.pick("all", exclude=exclude_channels, verbose=False)

    ## 滤波、陷波、重采样
    raw = raw.filter(1, 75, fir_design="firwin", verbose=False)
    raw = raw.notch_filter(50, verbose=False)
    raw.resample(256, npad="auto", verbose=False)

    event = mne.find_events(raw, stim_channel="Status", verbose=False)

    # According to the event intercept video before watching the previous section
    data = raw.get_data()
    del raw
    eeg_trial = data[:32, event[0, 0] : event[1, 0]]
    pps_trial = data[32:38, event[0, 0] : event[1, 0]]

    # # 外围生理信号所在通道，通道下标分别是ecg:32,33,34; gsr:40; resp:44; temp: 45
    # pps_channels = [32,33,34,40,44,45]
    # pps_trial = raw.get_data()[pps_channels, event[0, 0]:event[1, 0]]

    return eeg_trial.T, pps_trial.T


def get_eeg_pps_trials(subject_id):
    """
    Introduction:
        根据受试者编号获取受试者所有trial
    Args:
        subject_id: 受试者编号
        return: eeg_trials
    """
    bdf_path = "/data/MAHNOB/Sessions/{}/Part_{}_S_Trial{}_emotion.bdf"

    eeg_trial_list = []
    pps_trial_list = []

    for trail_index in range(2, 41, 2):

        eeg_pps_path = bdf_path.format(
            (subject_id - 1) * 130 + trail_index, subject_id, trail_index // 2
        )

        if not os.path.exists(eeg_pps_path):
            continue

        eeg_trial, pps_trial = _get_eeg_pps_trial(eeg_pps_path)

        eeg_trial_list.append(eeg_trial)
        pps_trial_list.append(pps_trial)

    return eeg_trial_list, pps_trial_list


def label_map(nb_classes, labels):
    """
    Introduction:
        根据sessions中获取到的labels数组映射标签
    Args:
        nb_classes: 映射类别数, 二分类或者三分类
        labels: 五个维度的标签列表, 分别是[arousal, valence, dominance, predictability, emotional]
    Return: arousal_label, valence_label. 如果返回label是-1, 则代表这条数据标签未找到对应映射, 应该删除该条数据
    """

    arousal_label = valence_label = -1

    if nb_classes == 2:

        if labels[0] > 5:
            arousal_label = 1
        else:
            arousal_label = 0

        if labels[1] > 5:
            valence_label = 1
        else:
            valence_label = 0

    elif nb_classes == 3:

        emo_map = {
            0: "Neutral",
            1: "Anger",
            2: "Disgust",
            3: "Fear",
            4: "Joy, Happiness",
            5: "Sadness",
            6: "Surprise",
            7: "Scream",
            8: "Bored",
            9: "Sleepy",
            10: "Unknown",
            11: "Amusement",
            12: "Anxiety",
        }

        emo_name = emo_map[labels[-1]]

        # 映射规则参考论文实现
        if emo_name in ["Sadness", "Disgust", "Neutral"]:
            arousal_label = 0  ## Calm
        elif emo_name in ["Joy, Happiness", "Amusement"]:
            arousal_label = 1  ## Medium
        elif emo_name in ["Surprise", "Fear", "Anger", "Anxiety"]:
            arousal_label = 2  ## Excited/Activated

        if emo_name in ["Fear", "Anger", "Disgust", "Sadness", "Anxiety"]:
            valence_label = 0  ## Unpleasant
        elif emo_name in ["Surprise", "Neutral"]:
            valence_label = 1  ## Neutal
        elif emo_name in ["Joy, Happiness", "Amusement"]:
            valence_label = 2  ## Pleasant

    return arousal_label, valence_label


def _get_label(label_path, nb_classes):
    """
    Introduction:
        根据label_path获取受试者标签
    Args:
        label_path: label 路径
        nb_classes: 映射类别数, 二分类或者三分类
        return: trial label
    """
    dom = parse(label_path)
    root = dom.documentElement
    if root.hasAttribute("feltEmo"):
        emotional = int(root.getAttribute("feltEmo"))
        arousal = int(root.getAttribute("feltArsl"))
        valence = int(root.getAttribute("feltVlnc"))
        dominance = int(root.getAttribute("feltCtrl"))
        predictability = int(root.getAttribute("feltPred"))
        # Store preprocessed results
        labels = np.array([arousal, valence, dominance, predictability, emotional])
        arousal_label, valence_label = label_map(nb_classes, labels)
    else:
        arousal_label, valence_label = -1, -1
    return arousal_label, valence_label


def get_labels(subject_id, nb_classes):
    """
    Introduction:
        根据受试者编号获取受试者所有trial的标签
    Args:
        subject_id: 受试者编号
        nb_classes: 映射类别数, 二分类或者三分类
        return: subject_labels
    """

    session_path = "/data/MAHNOB/Sessions/{}/session.xml"
    bdf_path = "/data/MAHNOB/Sessions/{}/Part_{}_S_Trial{}_emotion.bdf"
    tsv_path = tsv_path = (
        "/data/MAHNOB/Sessions/{}/P{}-Rec1-All-Data-New_Section_{}.tsv"
    )

    arousal_labels = []
    valence_labels = []
    for trail_index in range(2, 41, 2):
        # 这个session的眼动数据在3169、3170行有问题
        if subject_id == 25 and trail_index == 32:
            continue

        eeg_pps_path = bdf_path.format(
            (subject_id - 1) * 130 + trail_index, subject_id, trail_index // 2
        )
        eye_track_path = tsv_path.format(
            (subject_id - 1) * 130 + trail_index, subject_id, trail_index
        )
        label_path = session_path.format((subject_id - 1) * 130 + trail_index)

        if not (
            os.path.exists(eeg_pps_path)
            and os.path.exists(eye_track_path)
            and os.path.exists(label_path)
        ):
            continue

        # label_path = session_path.format((subject_id-1)*130 + trail_index)

        if not os.path.exists(label_path):
            continue
        arousal_label, valence_label = _get_label(label_path, nb_classes)
        arousal_labels.append(arousal_label)
        valence_labels.append(valence_label)
    return arousal_labels, valence_labels


def _get_eye_track_trials(eye_track_path):
    """
    Introduction:
        根据eye_track_path读取其眼动数据
    Args:
        eye_track_path: 待读取数据路径
        return: eye track trial
    """

    trial = pd.read_csv(eye_track_path, sep="\t", skiprows=23, on_bad_lines='skip')
    trial = trial[["PupilLeft", "PupilRight", "Event"]]
    trial = trial.loc[
        np.asarray(trial[trial["Event"] == "MovieStart"].index)[0] : np.asarray(
            trial[trial["Event"] == "MovieEnd"].index
        )[0]
        + 1
    ]
    trial = trial.drop(["Event"], axis=1)
    trial = np.asarray(trial)
    # data = data[~np.isnan(data).any(axis=1), :]
    trial[np.isnan(trial)] = -1

    return trial


def get_eye_track_trials(subject_id):
    """
    Introduction:
        根据受试者编号读取其眼动数据
    Args:
        subject_id: 受试者编号
        return: eye_track_trials
    """

    tsv_path = "/data/MAHNOB/Sessions/{}/P{}-Rec1-All-Data-New_Section_{}.tsv"

    eye_track_trial_list = []

    for trail_index in np.arange(2, 41, 2):

        eye_track_path = tsv_path.format(
            (subject_id - 1) * 130 + trail_index, subject_id, trail_index
        )

        if not os.path.exists(eye_track_path):
            continue

        eye_track_trial = _get_eye_track_trials(eye_track_path)
        eye_track_trial_list.append(eye_track_trial)

    return eye_track_trial_list


def get_subject_data(subject_id, nb_classes):
    """
    Introduction:
        根据受试者编号读取其eeg、眼动还有标签数据
    Args:
        subject_id: 受试者编号
        nb_classes: 映射类别数, 二分类或者三分类
        return: eeg_trials, pps_trials, eye_track_trials, arousal_labels, valence_labels
    """

    bdf_path = "/data/MAHNOB/Sessions/{}/Part_{}_S_Trial{}_emotion.bdf"
    tsv_path = "/data/MAHNOB/Sessions/{}/P{}-Rec1-All-Data-New_Section_{}.tsv"
    session_path = "/data/MAHNOB/Sessions/{}/session.xml"
    eeg_trials, pps_trials, eye_track_trials, arousal_labels, valence_labels = (
        [],
        [],
        [],
        [],
        [],
    )

    for trail_index in np.arange(2, 41, 2):
        # 这个session的眼动数据在3169、3170行有问题
        if subject_id == 25 and trail_index == 32:
            continue

        eeg_pps_path = bdf_path.format(
            (subject_id - 1) * 130 + trail_index, subject_id, trail_index // 2
        )
        eye_track_path = tsv_path.format(
            (subject_id - 1) * 130 + trail_index, subject_id, trail_index
        )
        label_path = session_path.format((subject_id - 1) * 130 + trail_index)

        if not (
            os.path.exists(eeg_pps_path)
            and os.path.exists(eye_track_path)
            and os.path.exists(label_path)
        ):
            continue

        arousal_label, valence_label = _get_label(label_path, nb_classes)
        # 数据不完整
        if arousal_label == -1 or valence_label == -1:
            continue

        eeg_trial, pps_trial = _get_eeg_pps_trial(eeg_pps_path)
        eye_track_trial = _get_eye_track_trials(eye_track_path)

        eeg_trials.append(eeg_trial)
        pps_trials.append(pps_trial)
        eye_track_trials.append(eye_track_trial)
        arousal_labels.append(arousal_label)
        valence_labels.append(valence_label)

    return eeg_trials, pps_trials, eye_track_trials, arousal_labels, valence_labels
