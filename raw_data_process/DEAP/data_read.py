import numpy as np
import pickle
import os


def get_labels(subject_id, threshold=5, data_path=None):
    path = os.path.join(data_path, "s{:0>2d}.dat")
    data = pickle.load(open(path.format(subject_id), "rb"), encoding="latin1")
    labels = data["labels"]
    arousal_labels = np.array(
        list(map(lambda x: 0 if x < threshold else 1, labels[:, 1]))
    )
    valence_labels = np.array(
        list(map(lambda x: 0 if x < threshold else 1, labels[:, 0]))
    )
    return arousal_labels, valence_labels


def get_eeg_data(subject_id, data_path=None):
    path = os.path.join(data_path, "s{:0>2d}.dat")
    data = pickle.load(open(path.format(subject_id), "rb"), encoding="latin1")
    eeg_data = np.array(data["data"][:, :32, 128 * 3 :])
    return eeg_data


def get_pps_data(subject_id, data_path=None):
    # https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
    path = os.path.join(data_path, "s{:0>2d}.dat")
    data = pickle.load(open(path.format(subject_id), "rb"), encoding="latin1")
    pps_data = np.array(data["data"][:, 32:, 128 * 3 :])
    return pps_data


def get_one_subject_data(subject_id, threshold=5, data_path=None):
    path = os.path.join(data_path, "s{:0>2d}.dat")
    data = pickle.load(open(path.format(subject_id), "rb"), encoding="latin1")
    eeg_data = np.array(data["data"][:, :32, 128 * 3 :])
    pps_data = np.array(data["data"][:, 32:, 128 * 3 :])
    labels = data["labels"]
    # arousal_labels = np.array(list(map(lambda x: 0 if x < threshold else 1, labels[:,1])))
    # valence_labels = np.array(list(map(lambda x: 0 if x < threshold else 1, labels[:,0])))

    arousal_labels = list(map(lambda x: 0 if x < threshold else 1, labels[:, 1]))
    valence_labels = list(map(lambda x: 0 if x < threshold else 1, labels[:, 0]))
    return eeg_data, pps_data, arousal_labels, valence_labels
