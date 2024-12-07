import os
import sys
from pathlib import Path
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from scipy import io
import copy
import torch


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()

from common.utils import find_nearest_folder, tensor_from_numpy


def initialize_graph(config, data_len, device):
    """初始化图数据"""
    adj, graph_indicator = createGraphStructer(config=config, batch_size=data_len)
    adj = normalization(adj).to(device)
    graph_indicator = tensor_from_numpy(graph_indicator, device)

    return (adj, graph_indicator)


def processing_adjacency(batch_size, ch_nums, save_path):
    single_adjacency = np.array([])
    i = 0

    while i < ch_nums:
        j = 0
        while j < ch_nums:
            temp = np.array([j, i])
            if len(single_adjacency) == 0:
                single_adjacency = temp[None]
            else:
                single_adjacency = np.append(single_adjacency, temp[None], axis=0)
            j += 1
        i += 1
    graph_num = 1
    adjacency_list = copy.copy(single_adjacency)
    while graph_num < batch_size:
        single_adjacency += ch_nums
        adjacency_list = np.append(adjacency_list, single_adjacency, axis=0)
        graph_num += 1

    io.savemat(save_path, {"adjacency_list": adjacency_list})
    print(f"Processing adjacency_list Done, Save to {save_path}")


def processing_weights(batch_size, ch_nums, save_path):
    pos_file = os.path.join(os.path.dirname(save_path), f"channels_pos_{ch_nums}.xlsx")
    pos = pd.read_excel(pos_file)
    if ch_nums == 62:
        signal = [
            [0, 2],
            [3, 4],
            [6, 12],
            [15, 21],
            [24, 30],
            [33, 39],
            [42, 48],
            [51, 55],
            [58, 60],
            [2, 0],
            [4, 3],
            [12, 6],
            [21, 15],
            [30, 24],
            [39, 33],
            [48, 42],
            [55, 51],
            [60, 58],
        ]
    elif ch_nums == 32:
        signal = [
            [0, 16],
            [1, 17],
            [4, 21],
            [8, 26],
            [13, 31],
            [16, 0],
            [17, 1],
            [21, 4],
            [26, 8],
            [31, 13],
        ]
    else:
        signal = [[0, 30], [4, 26], [9, 20], [14, 16]]

    result = []
    delta = 5  # 一个奇怪的系数，不知道有啥用

    for i in range(ch_nums):
        temp_x = [item / 10 for item in pos.iloc[i][1:4]]
        for j in range(ch_nums):
            temp_y = [item / 10 for item in pos.iloc[j][1:4]]
            temp = list(map(lambda x: (x[0] - x[1]) ** 2, zip(temp_x, temp_y)))
            if sum(temp) == 0:
                result.append(1)
            elif [i, j] in signal:
                result.append(min(1, delta / sum(temp)) - 1)
            else:
                result.append(min(1, delta / sum(temp)))

    tempnum, temp_result = 0, copy.copy(result)
    while tempnum < batch_size - 1:
        result += temp_result
        tempnum += 1
    result = np.array(result)
    io.savemat(save_path, {"adjacency_weight": result})
    print(f"Processing Weights Done, Save to {save_path}")


def createGraphStructer(config, batch_size):
    data_root = find_nearest_folder(config["data_path"])
    ch_nums = config["ch_nums"]
    os.makedirs(os.path.join(data_root, "adjacency"), exist_ok=True)
    adj_path = os.path.join(
        data_root,
        f"adjacency/adj_{batch_size}_{ch_nums}.mat",
    )
    weights_path = os.path.join(
        data_root,
        f"adjacency/weights_{batch_size}_{ch_nums}.mat",
    )

    if not os.path.exists(adj_path) or not os.path.exists(weights_path):
        print("Create Graph Structure")
        processing_adjacency(batch_size, ch_nums, adj_path)
        processing_weights(batch_size, ch_nums, weights_path)

    adjacency_list = io.loadmat(adj_path)["adjacency_list"]
    adjacency_weight = io.loadmat(weights_path)["adjacency_weight"][0]
    num_nodes = ch_nums
    graph_indicator = np.array([])
    i = 0
    while i < batch_size:
        temp = np.array([i] * num_nodes)
        graph_indicator = np.append(graph_indicator, temp)
        i += 1
    graph_indicator = graph_indicator.astype(np.int64)

    num_nodes = batch_size * num_nodes

    sparse_adjacency = sp.coo_matrix(
        (adjacency_weight, (adjacency_list[:, 0], adjacency_list[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    adjacency = sparse_adjacency.tocsr()
    return adjacency, graph_indicator


def normalization(adjacency):
    """compute L=D^-0.5 * (A+I) * D^-0.5,

    Args:
        adjacency: sp.csr_matrix.
    Returns:

        Adjacency matrix after normalization , torch.sparse.FloatTensor
    """
    # adjacency += sp.eye(adjacency.shape[0])    # add self-connection
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    # to torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency
