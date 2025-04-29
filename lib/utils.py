import pywt
import torch
import numpy as np

# log string
def log_string(log, string):
     # log为日志文件，string为字符串 log.flush为刷新日志缓冲区，确保刚刚写入的内容立刻保存到文件中
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
# 掩码作用并不是让 MAE 的值变成 1，而是
# 保证你直接对向量做平均时，能够准确地按有效元素来算加权后的平均值。
def metric(pred, label):
     # np.errstate忽略除零和无效错误
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
         # 掩码用于屏蔽标签为0的部分，防止除零错误，并将掩码转换为浮点数并标准化
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

# 计算掩码均方误差
# 掩码作用并不是让 MAE 的值变成 1，而是
# 保证你直接对向量做平均时，能够准确地按有效元素来算加权后的平均值。
# [1,0,2,4] -> [1,2,4]/3 = 2.3333
def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seq2instance(data, P, Q):
    # 将时间序列转会为监督学习的数据格式，将原始序列分割成输入子序列和对应的输出子序列。
    #P = y1，Q = y2，
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def disentangle(x, w, j):
    # 对输入张量进行小波分解，并将分解后的低频分量和高频分量重建出来
    # （10691，12，170，1）-》（10691，1，170，12）
    x = x.transpose(0,3,2,1) # [S,D,N,T]
    # 使用pywt.wavedec对张量x进行小波分型，得到个层次的系数coef ，w为小波类型，j为分解层次
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)

    return xl, xh

# main函数的self.mean, self.std, data = loadData(
#                                         self.traffic_file, self.input_len, self.output_len,
#                                         self.train_ratio, self.test_ratio, log)
def loadData(filepath, P, Q, train_ratio, test_ratio, log):
    # Traffic
    Traffic = np.load(filepath)['data'][...,:1]# Traffic = (17856,170,1)
    num_step = Traffic.shape[0]
    # 进行时间嵌入 TE = (17856,2)
    TE = np.zeros([num_step, 2])
    TE[:,1] = np.array([i % 288 for i in range(num_step)])
    TE[:,0] = np.array([(i // 288) % 7 for i in range(num_step)])
    # 扩展TE的索引值为1的维度（17856，1，2），np.repeat沿着新维度（axis:1）重复170次(17856.170.2)
    TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)
    log_string(log, f'Shape of data: {Traffic.shape}')
    # train/val/test 
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    trainData, trainTE = Traffic[: train_steps], TE_tile[: train_steps]
    valData, valTE = Traffic[train_steps : train_steps + val_steps], TE_tile[train_steps : train_steps + val_steps]
    testData, testTE = Traffic[-test_steps :], TE_tile[-test_steps :]
    # X, Y
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)
    trainXTE, trainYTE = seq2instance(trainTE, P, Q)
    valXTE, valYTE = seq2instance(valTE, P, Q)
    testXTE, testYTE = seq2instance(testTE, P, Q)
    # derive temporal embedding
    # trainXTE = (10691,12,170,2) trainYTE = (10691,12,170,2) trainYTE[0][0][0][1]==12
    trainTE = np.concatenate([trainXTE, trainYTE], axis=1)
    valTE = np.concatenate([valXTE, valYTE], axis=1)
    testTE = np.concatenate([testXTE, testYTE], axis=1)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)

    log_string(log, f'Shape of Train Data: {trainX.shape}')
    log_string(log, f'Shape of Validation Data: {valX.shape}')
    log_string(log, f'Shape of Test Data: {testX.shape}')

    log_string(log, f'Mean: {mean} & Std: {std}')
    
    return trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE, mean, std, trainData[...,0]