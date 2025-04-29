import gc
import os
import math
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from fastdtw import fastdtw
from .utils import log_string
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # 计算拉普拉斯矩阵
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    # 计算前k个最大特征值和特征向量，返回前k个特征值和对应的特征向量
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)
# 第一步：通过adj图计算它的拉普拉斯矩阵，然后通过拉普拉斯矩阵计算最大的k个特征值和特征向量，
# 然后返回这个特征值和特征向量
def get_eigv(adj,k):
    # 计算图的拉普拉斯矩阵的前k个特征值和特征向量
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def construct_tem_adj(data, num_node):
    # (train.data=10691.170.1)
    # 用于构建基于动态时间规整DTW距离的时间邻接矩阵-->这块可以拿来直接用
    # 按照天数分割数据集
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    # (288,170,1)
    data_mean = data_mean.squeeze().T
    # (170,288)
    dtw_distance = np.zeros((num_node, num_node))
    for i in tqdm(range(num_node)):
        # 使用tqdm库来显示进度条
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
            # [0]代表度量两个时间序列的相似度，值越小表示两个序列越相似
    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]
    nth = np.sort(dtw_distance.reshape(-1))[
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0]):
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0])+1] # NlogN edges
    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T).astype(int)
    return tem_matrix

def loadGraph(spatial_graph, temporal_graph, dims, data, log):
    # calculate spatial and temporal graph wavelets
    adj = np.load(spatial_graph)
    adj = adj + np.eye(adj.shape[0])
    if os.path.exists(temporal_graph):
        tem_adj = np.load(temporal_graph)
    else:
        tem_adj = construct_tem_adj(data, adj.shape[0])
        np.save(temporal_graph, tem_adj)
    spawave = get_eigv(adj, dims)
    temwave = get_eigv(tem_adj, dims)
    log_string(log, f'Shape of graphwave eigenvalue and eigenvector: {spawave[0].shape}, {spawave[1].shape}')

    # derive neighbors
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]

    log_string(log, f'Shape of localadj: {localadj.shape}')
    return localadj, spawave, temwave

# ----------------------------------------------------将上述代码改成周----------------------------------------------
def construct_weekly_tem_adj(data, num_node, tem_local_adj=True, radius=8):
    """
    基于每周典型日和DTW距离构建时序邻接，并返回全局/局部邻接及其聚合结果。

    Args:
        data (np.ndarray): 原始时序数据，shape=(T, num_node, 1) 或 (T, num_node)。
        num_node (int): 节点数量 N。
        tem_local_adj (bool): 是否返回每节点局部最相似排序索引。
        radius (int): fastdtw 搜索半径。

    Returns:
        tem_matrices (np.ndarray): shape=(7, N, N)，周一到周日的二值邻接矩阵。
        sorted_indices (np.ndarray or None): shape=(7, N, N)，每周每节点的局部相似度排序索引。
        avg_tem_matrix (np.ndarray): shape=(N, N)，7 天邻接矩阵逐元素求和除以7得到的平均邻接。
        final_sorted_indices (np.ndarray): shape=(N, N)，7 天排序索引之和的结果，再对每行进行 argsort 得到最终全局排序。
    """
    # 每天时间步数
    steps_per_day = 24 * 12
    num_days = data.shape[0] // steps_per_day

    # 1) 按 weekday 分组收集每日块
    weekday_slices = {w: [] for w in range(7)}
    for day in range(num_days):
        block = data[day*steps_per_day:(day+1)*steps_per_day]
        weekday = day % 7
        weekday_slices[weekday].append(block)

    # 2) 计算每个 weekday 的 "典型日" 平均序列，形状 (N, steps_per_day)
    data_mean = {}
    for w in range(7):
        arr = np.stack(weekday_slices[w], axis=0)       # (weeks, steps_per_day, N, 1)
        mean = np.mean(arr, axis=0)                     # (steps_per_day, N, 1)
        data_mean[w] = mean.squeeze().T                 # (N, steps_per_day)

    # 3) 初始化输出结构
    tem_matrices = np.zeros((7, num_node, num_node), dtype=int)
    sorted_indices = np.zeros((7, num_node, num_node), dtype=int) if tem_local_adj else None

    # 4) 分 weekday 计算 DTW 距离并生成邻接与局部排序
    for w in range(7):
        # 4.1) 距离矩阵
        dtw = np.zeros((num_node, num_node), dtype=float)
        for i in tqdm(range(num_node), desc=f"Weekday {w}"):
            for j in range(i, num_node):
                dist, _ = fastdtw(data_mean[w][i], data_mean[w][j], radius=radius)
                dtw[i, j] = dtw[j, i] = dist

        # 4.2) 局部相似度排序索引
        if tem_local_adj:
            md = dtw.copy()
            maxv = md.max()
            np.fill_diagonal(md, md.diagonal() + 2 * maxv)
            sorted_indices[w] = np.argsort(md, axis=1)
            del md; 
            gc.collect()

        # 4.3) 全局二值化阈值：保留前 N·log2N 条最相似边
        flat = dtw.ravel()
        idx = int(np.log2(num_node) * num_node)
        thresh = np.sort(flat)[idx]
        mat = (dtw <= thresh).astype(int)
        tem_matrices[w] = np.logical_or(mat, mat.T).astype(int)

    # 5) 叠加七天邻接矩阵并求平均
    avg_tem_matrix = np.sum(tem_matrices, axis=0) / 7.0

    # 6) 合并七天排序索引并全局排序
    if tem_local_adj:
        summed_sorted = np.sum(sorted_indices, axis=0)  # (N, N)
        final_sorted_indices = np.argsort(summed_sorted, axis=1)
        final_sorted_indices = (num_node - 1) - final_sorted_indices
    else:
        final_sorted_indices = None

    return tem_matrices, sorted_indices, avg_tem_matrix, final_sorted_indices


def loadGraph(spatial_graph, temporal_graph, temporal_local_graph, dims, data, log):
    """
    加载或构建图结构并计算其图小波基。

    Args:
        spatial_graph (str): 空间邻接矩阵 .npy 文件路径。
        temporal_graph (str): 保存或加载平均时间邻接(avg_tem_matrix)的 .npy 文件路径。
        temporal_local_graph (str): 保存或加载全局排序索引(final_sorted_indices)的 .npy 文件路径。
        dims (int): 特征维度 k。
        data (np.ndarray): 原始时序数据，用于生成时间邻接。
        log: 日志记录函数。

    Returns:
        avg_tem_matrix (np.ndarray): 平均时间邻接矩阵，shape=(N,N)。
        final_sorted_indices (np.ndarray): 全局排序索引，shape=(N,N)。
        spawave (tuple): (eigenvalues, eigenvectors) of spatial Laplacian.
        temwave (tuple): (eigenvalues, eigenvectors) of temporal Laplacian.
    """
    # 1) 加载空间邻接，并添加自环
    adj = np.load(spatial_graph)  # shape=(N,N)
    adj = adj + np.eye(adj.shape[0])

    # 2) 加载或构建平均时间邻接和全局排序索引
    if os.path.exists(temporal_graph) and os.path.exists(temporal_local_graph):
        avg_tem_matrix = np.load(temporal_graph)
        final_sorted_indices = np.load(temporal_local_graph)
    else:
        # construct_weekly_tem_adj 需要先导入
        _, _, avg_tem_matrix, final_sorted_indices = construct_weekly_tem_adj(
            data, adj.shape[0]
        )
        np.save(temporal_graph, avg_tem_matrix)
        np.save(temporal_local_graph, final_sorted_indices)

    # 3) 计算图小波基（Laplacian 特征值/向量）
    spawave = get_eigv(adj, dims)
    temwave = get_eigv(avg_tem_matrix, dims)
    log_string(log,
        f"Shape of spatial wave eigen: {spawave[0].shape}, {spawave[1].shape}" )
    log_string(log,
        f"Shape of temporal wave eigen: {temwave[0].shape}, {temwave[1].shape}" )

    # 4) 构建空间局部邻接(localadj) via 最短路径
    dis_sampled_nodes_number = int(math.log(adj.shape[0], 2)) * 2
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    localadj = np.argpartition(
        dist_matrix,
        dis_sampled_nodes_number,
        axis=1
    )[:, :dis_sampled_nodes_number]

    # 5) 记录及返回
    log_string(log, f"Shape of local adjacency: {localadj.shape}")
    return avg_tem_matrix, final_sorted_indices, localadj, spawave, temwave