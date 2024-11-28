import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch_scatter

import scipy.sparse as sp
import numpy as np


def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


def global_add_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_add(x, graph_indicator, dim=0, dim_size=num)


def top_rank(attention_score, graph_indicator, keep_ratio):
    """Based on the given attention_score, perform a pooling operation on each graph. pool each graph separately
    Arguments:
    ----------
        attention_score：torch.Tensor
            Attention score calculated using GCN，Z = GCN(A, X)
        graph_indicator：torch.Tensor
            Indicate which graph each node belongs to
        keep_ratio: float
            Proportion of nodes to keep ，the number is int(N * keep_ratio)
    """

    graph_id_list = list(set(graph_indicator.cpu().numpy()))
    mask = attention_score.new_empty((0,), dtype=torch.bool)
    impor = attention_score.new_empty((0,), dtype=torch.int64)
    for graph_id in graph_id_list:
        # print(graph_indicator == graph_id)
        graph_attn_score = attention_score[graph_indicator == graph_id]
        # print(graph_attn_score)
        # sys.exit()
        graph_node_num = len(graph_attn_score)
        graph_mask = attention_score.new_zeros((graph_node_num,), dtype=torch.bool)

        keep_graph_node_num = int(keep_ratio * graph_node_num)
        if len(graph_attn_score.size()) == 2:
            temp_gas, _ = graph_attn_score.sort(descending=True)
            _, sorted_index = temp_gas.sort(dim=0, descending=True)
            graph_mask[sorted_index[:, 0][:keep_graph_node_num]] = True
        else:
            _, sorted_index = graph_attn_score.sort(descending=True)
            graph_mask[sorted_index[:keep_graph_node_num]] = True
        mask = torch.cat((mask, graph_mask))
        impor = torch.cat((impor, sorted_index[: int(graph_node_num * keep_ratio)]))

    return mask, impor


def filter_adjacency(adjacency, mask):
    """Update the graph structure according to the mask

    Args:
        adjacency: torch.sparse.FloatTensor, Adjacency matrix before pooling
        mask: torch.Tensor(dtype=torch.bool), Node mask vector

    Returns:
        torch.sparse.FloatTensor, Normalized adjacency matrix after pooling
    """
    device = adjacency.device
    mask = mask.cpu().numpy()
    indices = adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices
    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]
    sparse_adjacency = sp.csr_matrix(
        (np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32
    )
    filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    return normalization(filtered_adjacency).to(device)


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


class FeatureGlobal(nn.Module):
    def __init__(self, input_dim, hidden_dim, tok):
        """feature extraction

        Args:
        ----
            input_dim: int, input dimension
            hidden_dim: int, hidden dimension
        """
        super(FeatureGlobal, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool = SelfAttentionPooling(hidden_dim * 3, tok)
        # print("tok:", tok)

    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))
        gcn3 = F.relu(self.gcn3(adjacency, gcn2))

        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)
        # pool, pool_graph_indicator, pool_adjacency = self.pool(adjacency, gcn_feature,
        #                                                        graph_indicator)
        pool, pool_graph_indicator, pool_adjacency, impor = self.pool(
            adjacency, gcn_feature, graph_indicator
        )
        readout = torch.cat(
            (
                global_avg_pool(pool, pool_graph_indicator),
                global_max_pool(pool, pool_graph_indicator),
            ),
            dim=1,
        )

        return readout, impor


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """Graph convolution ：L*X*\theta

        Args:
        ----------
            input_dim: int
                Dimension of node input feature
            output_dim: int
                Output feature dimension
            use_bias : bool, optional
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """Use sparse matrix multiplication"""
        if len(input_feature.shape) == 3:
            output_3d = torch.empty(
                input_feature.size(0),
                input_feature.size(1),
                self.output_dim,
                device="cuda",
            )
        if len(input_feature.shape) == 3:
            for band in range(input_feature.size(1)):
                support = torch.mm(input_feature[:, band, :], self.weight)
                output = torch.sparse.mm(adjacency, support)
                if self.use_bias:
                    output += self.bias
                output_3d[:, band, :] = output
        else:
            support = torch.mm(input_feature, self.weight)
            output = torch.sparse.mm(adjacency, support)
            if self.use_bias:
                output += self.bias
        return output_3d if len(input_feature.shape) == 3 else output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.input_dim)
            + " -> "
            + str(self.output_dim)
            + ")"
        )


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)

    def forward(self, adjacency, input_feature, graph_indicator):
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()

        attn_score = self.activation(attn_score)

        mask, impro = top_rank(attn_score, graph_indicator, self.keep_ratio)
        # mask = top_rank(attn_score, graph_indicator, self.keep_ratio)
        # print(mask.shape)
        if len(attn_score.size()) == 2:
            hidden_3d = torch.empty(
                int(input_feature.size(0) * self.keep_ratio),
                input_feature.size(1),
                input_feature.size(2),
                device="cuda",
            )
        if len(attn_score.size()) == 2:
            for i in range(attn_score.size(1)):
                hidden_3d[:, i, :] = input_feature[:, i, :][mask] * attn_score[:, i][
                    mask
                ].view(-1, 1)
        else:
            hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)

        # sys.exit()
        return (
            hidden_3d if len(attn_score.size()) == 2 else hidden,
            mask_graph_indicator,
            mask_adjacency,
            impro,
        )
