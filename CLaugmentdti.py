import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

# random.seed(0)


def aug_random_mask(input_feature, drop_percent=0.4):
    # input_feature = input_feature.detach()
    input_feature = torch.tensor(input_feature)
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    mask_idx = random.sample(node_idx, mask_num)

    for i in range(input_feature.shape[0]):
        # mask_idx = random.sample(node_idx, mask_num)

        for j in mask_idx:
            aug_feature[i][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.4):

    percent = drop_percent

    edge_num = len(input_adj)  # 9228 / 2
    add_drop_num = int(edge_num * percent)
    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)
    drop_idx.sort()
    drop_idx.reverse()
    for i in drop_idx:
        input_adj = np.delete(input_adj, i, axis=0)
    return input_adj






def aug_drop_node(input_fea, input_adj, drop_percent=0.5):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)  # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out


# if __name__ == "__main__":
    # main()
