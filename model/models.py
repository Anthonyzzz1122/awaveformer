import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
     # 目的是在一维卷积操作之后移除多余的填充维度
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size].contiguous()
    # 移除输入张量x的最后一个维度的多余部分
    # contiguous() 方法确保返回的张量在内存中是连续存储的，便于后续操作。


class TemEmbedding(nn.Module):
    def __init__(self, D):
        super(TemEmbedding, self).__init__()
        self.ff_te = FeedForward([295,D,D])
        # 表示输入维度295输出维度是D

    def forward(self, TE, T=288):
        '''
        TE: [B,T,2]
        return: [B,T,N,D]
        '''
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device) # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device) # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1) # [B,T,295]
        TE = TE.unsqueeze(dim=2) # [B,T,1,295]
        TE = self.ff_te(TE) # [B,T,1,F]

        return TE # [B,T,N,F]

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)
        # ln是一个归一化层

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

class Sparse_Spatial_Attention(nn.Module):
    def __init__(self, heads, dims, samples, localadj, mask1, mask2):
        super(Sparse_Spatial_Attention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims
        self.s = samples

        # 将局部邻接矩阵保存为 tensor（用于局部注意力阶段）
        self.la = torch.as_tensor(localadj, dtype=torch.long)  # 局部邻接表，每个节点的近邻索引列表
        # 用于局部注意力得分投影：把 L 维降成 1
        self.proj = nn.Linear(self.la.shape[1], 1)
        # 将 mask1 和 mask2 保存为 tensor，并对 mask2 做归一化
        mask1_tensor = torch.tensor(mask1, dtype=torch.float32)
        mask2_int = torch.tensor(mask2, dtype=torch.long)  # mask2 原始整数排名矩阵
        N = mask2_int.shape[0]
        mask2_norm = mask2_int.float() / float(N - 1)      # 将排名归一化到 [0,1]

        # 为每个注意力头生成各自的 mask2 权重矩阵（关注不同 top-K 强相关节点）
        mask2_head = torch.zeros((self.h, N, N), dtype=torch.float32)
        for i in range(self.h):
            # 计算当前头需要保留的强相关节点数量 topK_i（按相关性排序取前 topK_i 个）
            topK_i = int(N * (i + 1) / self.h)            # 随着 i 增大，topK_i 增加
            if topK_i < 1:
                topK_i = 1
            if topK_i > N:
                topK_i = N
            threshold = N - topK_i                       # 排名阈值：仅保留排名 >= threshold 的节点对
            # 生成当前头的 mask2 权重：高于阈值的相关节点保留归一化权重，其他置零
            mask2_head[i] = mask2_norm * (mask2_int >= threshold).float()

        # 将 mask1 和每头的 mask2 保存为缓冲区，以便在 forward 中使用（不会更新梯度）
        self.register_buffer('mask1', mask1_tensor)
        self.register_buffer('mask2_big', mask2_head.view(-1, N))  # 展开 head 维度后的 mask2，shape=(h*N, N)

        # 定义 Q, K, V 的投影层以及输出层
        self.qfc = nn.Linear(features, features)
        self.kfc = nn.Linear(features, features)
        self.vfc = nn.Linear(features, features)
        self.ofc = nn.Linear(features, features)
        # 残差层归一化和前馈网络
        self.ln = nn.LayerNorm(features)
        self.ff = nn.Linear(features, features)  # 简化：这里假设 FeedForward([features,features,features], True) 等价于线性层

    def forward(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec,IsMask1, IsMask2):
        """
        x: [B, T, N, D]
        spa_eigvalue, tem_eigvalue: [D]    (空间/时间拉普拉斯特征值)
        spa_eigvec, tem_eigvec: [N, D]     (空间/时间拉普拉斯特征向量)
        return: [B, T, N, D]
        """
        B, T, N, D = x.shape

        # 将时空特征加入输入（残差连接）
        x_ = x + torch.matmul(spa_eigvec, torch.diag_embed(spa_eigvalue)) \
               + torch.matmul(tem_eigvec, torch.diag_embed(tem_eigvalue))

        # 计算多头注意力的 Q, K, V 表示
        Q = self.qfc(x_)  # [B, T, N, features]
        K = self.kfc(x_)
        V = self.vfc(x_)

        # 将 Q, K, V 按注意力头分割并在批次维度拼接，以便并行计算多头注意力
        Q = torch.cat(torch.split(Q, self.d, dim=-1), dim=0)  # [B*h, T, N, d]
        K = torch.cat(torch.split(K, self.d, dim=-1), dim=0)  # [B*h, T, N, d]
        V = torch.cat(torch.split(V, self.d, dim=-1), dim=0)  # [B*h, T, N, d]

        BH, T, N, d = K.shape  # 注意：此时 BH = B * self.h
        original_B = BH // self.h

        # 局部注意力阶段：利用局部邻接 self.la 为每个节点选取近邻进行注意力估计
        # 计算每个节点对其局部邻居的 QK 相似度（用于筛选全局注意力的 Query 节点）
        K_expand = K.unsqueeze(-3).expand(BH, T, N, N, d)                    # 扩展 K 以便与每个节点的邻居进行点积
        # 使用 self.la 提供的每个节点的邻居索引，采样每个节点的局部 K
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.la, :]  # [BH, T, N, L, d] L为局部邻居数
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # [BH, T, N, L]

        # 将每个节点对其邻居的注意力得分经过线性层投影，得到每个节点的“全局相关性”分数 M
        M = self.proj(Q_K_sample).squeeze(-1)  # [BH, T, N]，每个节点对应一个分数
        # 从每个时间步中选取得分最高的若干节点（全局注意力的 Query 节点集合）
        Sampled_Nodes = int(self.s * math.log(N, 2))
        M_top = M.topk(Sampled_Nodes, dim=-1, sorted=False)[1]  # [BH, T, Sampled_Nodes] 注意力得分最高的节点索引

        # 全局注意力阶段：使用筛选出的节点 (M_top) 作为 Query，对所有 Key 执行全局注意力计算
        Q_reduce = Q[torch.arange(BH)[:, None, None],
                     torch.arange(T)[None, :, None],
                     M_top, :]                      # 选取每个批次每个时间步中 TopK 节点对应的 Q，shape=[BH, T, Sampled_Nodes, d]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))      # 计算选出 Query 节点与所有节点之间的 QK，shape=[BH, T, Sampled_Nodes, N]
        Q_K = Q_K / (d ** 0.5)                                # 缩放注意力得分

        # 应用软掩码 mask1 和 mask2：在 softmax 之前调整全局注意力得分
        # 根据选取的 Query 节点 (M_top) 提取对应的 mask1 和 mask2 权重值
        M_top_flat = M_top.reshape(-1)  # 展平索引列表，长度 L = BH * T * Sampled_Nodes
        # 为 mask1 提取对应行（无需考虑注意力头差异）
        mask1_vals = self.mask1[M_top_flat]                   # [L, N] 对应每个选中节点 i 的 mask1 行
        # 计算每个样本对应的注意力头偏移，用于从 mask2_big 提取（将 head 和节点索引合并）
        head_ids = torch.arange(self.h, device=x.device).repeat_interleave(original_B)
        head_offsets = (head_ids * N).view(BH, 1, 1)           # [BH,1,1] 每个批次样本对应的 mask2 偏移
        M_top_head = M_top + head_offsets                     # 将注意力头偏移加到节点索引上
        mask2_vals = self.mask2_big[M_top_head.reshape(-1)]   # [L, N] 从展开的 mask2_big 提取对应行
        # 将 mask1 和 mask2 权重乘到 Q_K 上（逐元素相乘）
        if(IsMask1):
            mask1_vals = mask1_vals.view(BH, T, Sampled_Nodes, N)
            Q_K = Q_K * mask1_vals                   # 调整后的全局注意力 QK 得分
        if(IsMask2):
            mask2_vals = mask2_vals.view(BH, T, Sampled_Nodes, N)
            Q_K = Q_K * mask2_vals                   # 调整后的全局注意力 QK 得分

        # 计算注意力权重并应用于 V
        attn = torch.softmax(Q_K, dim=-1)                     # 对所有 Key 计算 softmax 注意力权重
        # 拷贝机制：为每个原始节点选择得分最高的 Query（索引 cp），并从对应 Query 的输出中拷贝值
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)  # [BH, T, N, 1] 每个节点在 Sampled_Nodes 维度上得分最高的 Query 索引
        # 根据 cp 从值矩阵中选取对应 Query 的输出作为每个节点的值
        value = torch.matmul(attn, V)  # [BH, T, Sampled_Nodes, d] 计算选定的 Query 对应的加权值
        value = value.unsqueeze(-3).expand(BH, T, N, Sampled_Nodes, d)[
            torch.arange(BH)[:, None, None, None],
            torch.arange(T)[None, :, None, None],
            torch.arange(N)[None, None, :, None],
            cp, :
        ].squeeze(-2)  # [BH, T, N, d] 每个节点选取对应 Query 的值

        # 将多头的输出拼接回原始维度
        value = torch.cat(torch.split(value, original_B, dim=0), dim=-1)  # [B, T, N, features] 将各头输出在特征维拼接
        value = self.ofc(value)       # 输出投影
        value = self.ln(value)        # LayerNorm 规范化

        return self.ff(value)         # 残差前馈网络输出最终

    
class TemporalAttention(nn.Module):
    def __init__(self, heads, dims):
        super(TemporalAttention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qfc = FeedForward([features,features])
        self.kfc = FeedForward([features,features])
        self.vfc = FeedForward([features,features])
        self.ofc = FeedForward([features,features])
        
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features,features,features], True)

    def forward(self, x, te, Mask=True):
        '''
        x: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        x += te

        query = self.qfc(x) #[B,T,N,F]
        key = self.kfc(x) #[B,T,N,F]
        value = self.vfc(x) #[B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3) # [k*B,T,N,d]
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1) # [k*B,N,d,T]
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3) # [k*B,N,T,d]

        attention = torch.matmul(query, key) # [k*B,N,T,T]
        attention /= (self.d ** 0.5) # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attention).to(x.device) # [k*B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1) # [k*B,N,T,T]

        value = torch.matmul(attention, value) # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0]//self.h, 0), -1).permute(0,2,1,3) # [B,T,N,F]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value)

class TemporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, xh):
        xh = self.tcn(xh.transpose(1,3)).transpose(1,3)
        return xh
    
class Dual_Enconder(nn.Module):
    def __init__(self, heads, dims, samples, localadj, avg_tem_matrix, final_sorted_indices,spawave, temwave):
        super(Dual_Enconder, self).__init__()
        # self.temporal_conv = TemporalConvNet(heads * dims)
        self.temporal_att = MyTemporalAttention(heads, dims)
        self.temporal_conv = ImprovedTemporalConvNet(heads*dims)

        self.spatial_att_l = Sparse_Spatial_Attention(heads, dims, samples, localadj,avg_tem_matrix, final_sorted_indices)
        self.spatial_att_h = Sparse_Spatial_Attention(heads, dims, samples, localadj,avg_tem_matrix, final_sorted_indices)
        
        spa_eigvalue = torch.from_numpy(spawave[0].astype(np.float32))
        self.spa_eigvalue = nn.Parameter(spa_eigvalue, requires_grad=True)        
        self.spa_eigvec = torch.from_numpy(spawave[1].astype(np.float32))

        tem_eigvalue = torch.from_numpy(temwave[0].astype(np.float32))
        self.tem_eigvalue = nn.Parameter(tem_eigvalue, requires_grad=True)        
        self.tem_eigvec = torch.from_numpy(temwave[1].astype(np.float32))
        
    def forward(self, xl, xh, te):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        xl = self.temporal_att(xl, te) # [B,T,N,F]
        xh = self.temporal_conv(xh) # [B,T,N,F]
        
        spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec.to(xl.device), self.tem_eigvalue, self.tem_eigvec.to(xl.device),False,False) # [B,T,N,F]
        spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec.to(xl.device), self.tem_eigvalue, self.tem_eigvec.to(xl.device),False,False) # [B,T,N,F]
        xl = spa_statesl + xl
        xh = spa_statesh + xh
        
        return xl, xh

class Adaptive_Fusion(nn.Module):
    def __init__(self, heads, dims):
        super(Adaptive_Fusion, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qlfc = FeedForward([features,features])
        self.khfc = FeedForward([features,features])
        self.vhfc = FeedForward([features,features])
        self.ofc = FeedForward([features,features])
        
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features,features,features], True)

    def forward(self, xl, xh, te, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        xl += te
        xh += te

        query = self.qlfc(xl) # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh)) # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh)) # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d,-1), 0).permute(0,2,1,3) # [k*B,N,T,d]
        keyh = torch.cat(torch.split(keyh, self.d,-1), 0).permute(0,2,3,1) # [k*B,N,d,T]
        valueh = torch.cat(torch.split(valueh, self.d,-1), 0).permute(0,2,1,3) # [k*B,N,T,d]

        attentionh = torch.matmul(query, keyh) # [k*B,N,T,T]
        
        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attentionh).to(xl.device) # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)
        
        attentionh /= (self.d ** 0.5) # scaled
        attentionh = F.softmax(attentionh, -1) # [k*B,N,T,T]

        value = torch.matmul(attentionh, valueh) # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0]//self.h, 0), -1).permute(0,2,1,3) # [B,T,N,F]
        value = self.ofc(value)
        value = value + xl

        value = self.ln(value)

        return self.ff(value)
    
class STWave(nn.Module):
    def __init__(self, heads,
                dims, layers, samples,
                localadj,avg_tem_matrix, final_sorted_indices,spawave, temwave,
                input_len, output_len,
                adaptive_embedding_dim):
        
        super(STWave, self).__init__()
        features = heads * dims
        I = torch.arange(localadj.shape[0]).unsqueeze(-1)
        localadj_full = torch.cat([I, torch.from_numpy(localadj)], -1)
        self.input_len = input_len

        self.dual_enc = nn.ModuleList([Dual_Enconder(heads, dims, samples, localadj_full,avg_tem_matrix, final_sorted_indices, spawave, temwave) for i in range(layers)])
        self.adp_f = Adaptive_Fusion(heads, dims)
        
        self.pre_l = nn.Conv2d(input_len, output_len, (1,1))
        self.pre_h = nn.Conv2d(input_len, output_len, (1,1))
        self.pre = nn.Conv2d(input_len, output_len, (1,1))

        self.start_emb_l = FeedForward([1, features, features])
        self.start_emb_h = FeedForward([1, features, features])
        self.end_emb = FeedForward([features, features, 1])
        self.end_emb_l = FeedForward([features, features, 1])
        self.te_emb = TemEmbedding(features)
        self.apt_emb = Adaptive_Embedding(adaptive_embedding_dim, heads, dims)

    def forward(self, XL, XH, TE):
        '''
        XL: [B,T,N,F]
        XH: [B,T,N,F]
        TE: [B,T,2]
        return: [B,T,N,1]
        '''
        xl, xh = self.start_emb_l(XL), self.start_emb_h(XH)
        te = self.te_emb(TE)
        # xl, xh = self.apt_emb(xl, xh, te)
        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:,:self.input_len,:,:])
        
        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)
        hat_y = self.adp_f(hat_y_l, hat_y_h, te[:,self.input_len:,:,:])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)
        
        return hat_y, hat_y_l
    
# -----------------------------------------------------------------------
class Adaptive_Embedding(nn.Module):
    def __init__(self, adaptive_embedding_dim, heads, dims):
        '''
        XL: [B,T,N,F]
        XH: [B,T,N,F]
        TE: [B,T,N,F]
        return: [B,T,N,F]
        '''
        super(Adaptive_Embedding, self).__init__()
        features = heads * dims
        self.adaptive_embedding_dim = adaptive_embedding_dim
        # 修正 self.ofc 的输入维度
        self.ofc = nn.Linear(features + adaptive_embedding_dim + features, features)

    def forward(self, xl, xh, te):
        te = te[:, :xl.shape[1], :, :]
        te = te.repeat(1, 1, xl.shape[2], 1)
        features_l = [xl]
        features_h = [xh]
        features_l.append(te)

        if self.adaptive_embedding_dim > 0:
            # 对张量进行初始化，仅作用于 xl
            adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(xl.shape[1], xl.shape[2], self.adaptive_embedding_dim))
            ).to(xl.device)
            adp_emb = adaptive_embedding.expand(
                size=(xl.shape[0], *adaptive_embedding.shape)
            ).to(xl.device)
            features_l.append(adp_emb)

        xl_cat = torch.cat(features_l, dim=-1).to(xl.device)  # (batch_size, in_steps, num_nodes, model_dim)
        xh_cat = torch.cat(features_h, dim=-1).to(xl.device)  # 保持 xh 不受 adaptive_embedding 影响
        emb_result_l = (self.ofc(xl_cat) + xl).to(xl.device)
        emb_result_h = xh  # 不对 xh 进行额外处理
        return emb_result_l, emb_result_h
# ---------------------------------------------------------------------------
class MyTemporalAttention(nn.Module):
    def __init__(self, heads, dims):
        super(MyTemporalAttention, self).__init__()
        self.h = heads       # 注意力头数
        self.d = dims        # 每个头的维度
        features = heads * dims

        # Q, K, V 和输出 O 的前馈映射
        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        # LayerNorm + FFN
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, x, te, Mask=True, Mask2=True):
        """
        x: [B, T, N, F]
        te: [B, T, N, F]
        Mask: 是否启用因果掩码
        Mask2: 是否启用改进型掩码
        return: [B, T, N, F]
        """
        B, T, N, D = x.shape
        # 注入时间编码
        x = x + te

        # 1) 生成 Q, K, V
        query = self.qfc(x)  # [B,T,N,features]
        key   = self.kfc(x)
        value = self.vfc(x)

        # 2) 拆头：按 feature 维切出 h 份，再在 batch 维 concat
        q = torch.cat(query.split(self.d, dim=-1), dim=0)  # [h*B, T, N, d]
        k = torch.cat(key.split(self.d,   dim=-1), dim=0)
        v = torch.cat(value.split(self.d, dim=-1), dim=0)

        # 3) 调整维度以便做注意力
        q = q.permute(0, 2, 1, 3)  # [h*B, N, T, d]
        k = k.permute(0, 2, 3, 1)  # [h*B, N, d, T]
        v = v.permute(0, 2, 1, 3)  # [h*B, N, T, d]

        # 4) 计算打分并缩放
        attn = torch.matmul(q, k) / math.sqrt(self.d)  # [h*B, N, T, T]

        # 5) 因果掩码（Mask1）
        if Mask:
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            mask1 = causal.unsqueeze(0).unsqueeze(0)        # [1,1,T,T]
            mask1 = mask1.expand(self.h * B, N, T, T)      # [h*B, N, T, T]
            attn = attn.masked_fill(~mask1, float('-inf'))

        # 6) 改进型掩码（Mask2）：头特定时段为主，也关注其他时间段
        if Mask2:
            # reshape 回头维度
            attn_heads = attn.view(self.h, B, N, T, T)  # [h, B, N, T, T]
            # 每个头关注不同时间段，但保留其他段一定比例的信息
            segment = T // self.h
            beta = 0.3  # 跨段信息保留权重
            for i in range(self.h):
                head_attn = attn_heads[i]  # [B, N, T, T]
                start = i * segment
                end = (i + 1) * segment if i < self.h - 1 else T
                # 布尔掩码：True 表示本头重点段
                focus_mask = torch.zeros((T, T), device=x.device, dtype=torch.bool)
                focus_mask[:, start:end] = True
                focus_mask = focus_mask.unsqueeze(0).unsqueeze(0).expand(B, N, T, T)
                # 在重点段保持原值，在其他段乘以 beta
                head_attn = torch.where(focus_mask,
                                        head_attn,
                                        head_attn * beta)
                attn_heads[i] = head_attn
            # flatten 回原 shape
            attn = attn_heads.view(self.h * B, N, T, T)

        # 7) Softmax
        attn = F.softmax(attn, dim=-1)

        # 8) 加权求和值并合并头
        out = torch.matmul(attn, v)                     # [h*B, N, T, d]
        out = out.permute(0, 2, 1, 3).contiguous()       # [h*B, T, N, d]
        out = torch.cat(out.split(B, dim=0), dim=-1)     # [B, T, N, h*d]

        # 9) 输出映射 + 残差 + LayerNorm + FFN
        out = self.ofc(out)
        out = out + x
        out = self.ln(out)
        return self.ff(out)

# ------------------------------优化-------------------------------   
class GatedTemporalBlock(nn.Module):
    """
    A single residual gated temporal block (WaveNet style) for high-frequency irregularity.
    Uses two parallel convolutions: filter (tanh) and gate (sigmoid).
    Input/Output: x: [B*N, C, T]
    """
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        # filter conv
        self.conv_filter = weight_norm(
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=padding)
        )
        # gate conv
        self.conv_gate = weight_norm(
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=padding)
        )
        self.chomp = Chomp1d(padding)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # residual/skip not changing channels
    
    def forward(self, x):
        # x: [B*N, C, T]
        # filter branch
        f = self.conv_filter(x)
        f = self.chomp(f)
        f = torch.tanh(f)
        # gate branch
        g = self.conv_gate(x)
        g = self.chomp(g)
        g = torch.sigmoid(g)
        # gated output
        out = f * g
        out = self.dropout(out)
        # residual connection
        return self.relu(out + x)

class ImprovedTemporalConvNet(nn.Module):
    """
    Stack of gated temporal blocks for high-frequency component.
    Input: xh [B, T, N, F], output same.
    """
    def __init__(self, features, kernel_size=3, dropout=0.2, levels=1):
        super().__init__()
        layers = []
        # flatten batch and nodes dims inside forward
        for i in range(levels):
            dilation = 2 ** i
            layers.append(
                GatedTemporalBlock(
                    channels=features,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
    
    def forward(self, xh):
        # xh: [B, T, N, F]
        B, T, N, F = xh.shape
        # merge B and N dims, permute to [B*N, F, T]
        x = xh.permute(0,2,3,1).reshape(B*N, F, T)
        # apply gated temporal convs
        out = self.network(x)
        # reshape back
        out = out.view(B, N, F, T).permute(0,3,1,2).contiguous()
        return out
#-----------------------------STWAVE COPY---------------------------------------------------
class STWaveCOPY(nn.Module):
    # def __init__(self, heads,
    #             dims, layers, samples,
    #             localadj,spawave, temwave,
    #             input_len, output_len,
    #             adaptive_embedding_dim):
    def __init__(self, heads,
                dims, layers, samples,
                localadj,spawave, temwave,
                input_len, output_len,
                adaptive_embedding_dim):
        
        super(STWaveCOPY, self).__init__()
        features = heads * dims
        I = torch.arange(localadj.shape[0]).unsqueeze(-1)
        localadj_full = torch.cat([I, torch.from_numpy(localadj)], -1)
        self.input_len = input_len

        self.dual_enc = nn.ModuleList([Dual_Enconder(heads, dims, samples, localadj_full, spawave, temwave) for i in range(layers)])
        self.adp_f = Adaptive_Fusion(heads, dims)
        
        self.pre_l = nn.Conv2d(input_len, output_len, (1,1))
        self.pre_h = nn.Conv2d(input_len, output_len, (1,1))
        self.pre = nn.Conv2d(input_len, output_len, (1,1))

        self.start_emb_l = FeedForward([1, features, features])
        self.start_emb_h = FeedForward([1, features, features])
        self.end_emb = FeedForward([features, features, 1])
        self.end_emb_l = FeedForward([features, features, 1])
        self.te_emb = TemEmbedding(features)
        self.apt_emb = Adaptive_Embedding(adaptive_embedding_dim, heads, dims)

    def forward(self, XL, XH, TE):
        '''
        XL: [B,T,N,F]
        XH: [B,T,N,F]
        TE: [B,T,2]
        return: [B,T,N,1]
        '''
        xl, xh = self.start_emb_l(XL), self.start_emb_h(XH)
        te = self.te_emb(TE)
        # xl, xh = self.apt_emb(xl, xh, te)
        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:,:self.input_len,:,:])
        
        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)
        hat_y = self.adp_f(hat_y_l, hat_y_h, te[:,self.input_len:,:,:])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)
        
        return hat_y, hat_y_l
class Sparse_Spatial_AttentionCOPY(nn.Module):
    def __init__(self, heads, dims, samples, localadj,avg_tem_matrix, final_sorted_indices):
        super(Sparse_Spatial_AttentionCOPY, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims
        self.s = samples
        self.la = localadj

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])
        
        self.ln = nn.LayerNorm(features)
        self.ff = FeedForward([features,features,features], True)
        self.proj = nn.Linear(self.la.shape[1], 1)

    def forward(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec,mask1=False, mask2=False):
        '''
        x: [B,T,N,D]
        spa_eigvalue, tem_eigvalue: [D]
        spa_eigvec, tem_eigvec: [N,D]
        return: [B,T,N,D]
        '''
        x_ = x + torch.matmul(spa_eigvec, torch.diag_embed(spa_eigvalue)) + torch.matmul(tem_eigvec, torch.diag_embed(tem_eigvalue))

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        Q = torch.cat(torch.split(Q,self.d,-1), 0)
        K = torch.cat(torch.split(K,self.d,-1), 0)
        V = torch.cat(torch.split(V,self.d,-1), 0)

        B, T, N, D = K.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.la, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        Sampled_Nodes = int(self.s * math.log(N, 2))
        M = self.proj(Q_K_sample).squeeze(-1)
        M_top = M.topk(Sampled_Nodes, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(T)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        Q_K /= (self.d ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)
        
        # copy operation
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2,-1)

        value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, M_top.shape[-1], V.shape[-1])[torch.arange(B)[:, None, None, None],
                                                                                                 torch.arange(T)[None, :, None, None],
                                                                                                 torch.arange(N)[None, None, :, None],cp,:].squeeze(-2)

        value = torch.cat(torch.split(value, value.shape[0]//self.h, 0), -1)
        value = self.ofc(value)

        value = self.ln(value)

        return self.ff(value)