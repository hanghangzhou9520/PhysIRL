import torch

def MPJPE(V_pred,V_trgt):#V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z}
    return torch.linalg.norm(V_trgt- V_pred,dim=-1).mean()
def MPJPE_nodeandpose(V_pred,V_trgt):#V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z}
    a = torch.linalg.norm(V_trgt- V_pred,dim=-1)
    node_erro = a.mean(dim=0)
    node_erro = node_erro.mean(dim=0)
    pose_erro = a.mean(dim=0)
    pose_erro = pose_erro.mean(dim=1)
    return node_erro*1000,pose_erro*1000


def MPJPE_timelimit_all(V_pred, V_trgt,
                        time_limit):  # V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME,,time_limit=where to evaluate in time, Points,{x,y,z}
    T = V_pred.shape[1]
    T = max(int(T * time_limit), 1)
    return torch.linalg.norm(V_trgt[:, :T, ...] - V_pred[:, :T, ...], dim=-1)


def MPJPE_timelimit(V_pred, V_trgt,
                    time_limit):  # V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME,,time_limit=where to evaluate in time, Points,{x,y,z}
    T = V_pred.shape[1]
    T = max(int(T * time_limit), 1)
    meann = torch.linalg.norm(V_trgt[:, :T, ...] - V_pred[:, :T, ...], dim=-1).mean()

    return meann
def MPJPE_torso(V_pred,V_trgt,torso_joint = 13):  #V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z}, torso_joint= for specific path

    return torch.linalg.norm(V_trgt[:,:,torso_joint,:]- V_pred[:,:,torso_joint,:],dim=-1).mean()


def MPJPE_timelimit_all(V_pred, V_trgt,
                        time_limit):  # V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME,,time_limit=where to evaluate in time, Points,{x,y,z}
    T = V_pred.shape[1]
    T = max(int(T * time_limit), 1)
    return torch.linalg.norm(V_trgt[:, :T, ...] - V_pred[:, :T, ...], dim=-1)


def MPJPE_timelimit(V_pred, V_trgt,
                    time_limit):  # V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME,,time_limit=where to evaluate in time, Points,{x,y,z}
    T = V_pred.shape[1]
    T = max(int(T * time_limit), 1)
    meann = torch.linalg.norm(V_trgt[:, :T, ...] - V_pred[:, :T, ...], dim=-1).mean()

    return meann


def MPJPE_torso_timelimit(V_pred, V_trgt, time_limit,
                          torso_joint=13):  # V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z},time_limit=where to evaluate in time, torso_joint= for specific path
    T = V_pred.shape[1]
    T = max(int(T * time_limit), 1)
    # 13 is the center of the torso in GTA IM
    return torch.linalg.norm(V_trgt[:, :T, torso_joint, :] - V_pred[:, :T, torso_joint, :], dim=-1).mean()
def diff_cen(x, dim):
    h = 1
    center_diff = (x[:, 2:] - x[:, :-2]) / (2 * h)
    # 补齐两端
    prepend = (x[:, 1:2] - x[:, :1]) / h  # 前向差分
    append = (x[:, -1:] - x[:, -2:-1]) / h  # 后向差分
    center_diff = torch.cat((prepend, center_diff, append), dim=dim)
    return center_diff

def ECLoss(V_pred, V_trgt, m):
    """
    包含位置误差、动量守恒和动能守恒的物理约束损失函数

    参数:
        V_pred (torch.Tensor): 预测位置张量，形状 (128, 10, 21, 3)
        V_trgt (torch.Tensor): 真实位置张量，形状 (128, 10, 21, 3)
        m (torch.Tensor): 质量张量，形状 (128, 5, 21, 21)

    返回:
        torch.Tensor: 组合物理损失
    """
    # 1. 质量处理: 沿批次和时间维度求平均 -> (21,)
    m_mean = torch.mean(m, dim=(0,1, 3))  # 平均后形状为 (21,)
    # 调整形状以便广播: (1, 1, 21, 1)
    m_mean = m_mean.view(1, 1, 21, 1)
    posipre = m_mean * V_pred
    posite = m_mean * V_trgt

    # 3. 速度计算 (时间步差分)
    pred_vel = V_pred[:, 1:, :, :] - V_pred[:, :-1, :, :] # (128, 9, 21, 3)
    trgt_vel = V_trgt[:, 1:, :, :] - V_trgt[:, :-1, :, :]  # (128, 9, 21, 3)

    # 4. 动量损失 (系统总动量守恒)
    # 4.1 计算每个粒子的动量: p = m * v
    pred_momentum = m_mean * pred_vel  # (128, 9, 21, 3)
    trgt_momentum = m_mean * trgt_vel  # (128, 9, 21, 3)
    #
    # # 4.2 计算系统总动量 (所有粒子求和)
    # pred_total_momentum = torch.sum(pred_momentum, dim=2)  # (128, 9, 3)
    # trgt_total_momentum = torch.sum(trgt_momentum, dim=2)  # (128, 9, 3)
    #
    # # 4.3 动量守恒损失 (系统总动量变化)
    # momentum_loss = torch.mean(torch.linalg.norm(
    #     pred_total_momentum - trgt_total_momentum,
    #     ord=2,
    #     dim=-1
    # ) ** 2)

    # 5. 动能损失 (系统总动能守恒)
    # 5.1 计算每个粒子的动能: K = 0.5 * m * v^2
    pred_kinetic = 0.5 * m_mean * pred_vel ** 2
    trgt_kinetic = 0.5 * m_mean * trgt_vel ** 2
    #
    # # 5.2 计算系统总动能 (所有粒子求和)
    # pred_total_kinetic = torch.sum(pred_kinetic, dim=2)  # (128, 9, 1)
    # trgt_total_kinetic = torch.sum(trgt_kinetic, dim=2)  # (128, 9, 1)
    #
    # # 5.3 动能守恒损失 (系统总动能变化)
    #

    # 6. 组合损失 (可调整权重)
    MPJPE = torch.linalg.norm(V_trgt- V_pred,dim=-1).mean()
    #MPJPE1 = torch.linalg.norm(trgt_vel - pred_vel, dim=-1).mean()
    #pos_loss = torch.linalg.norm(posite- posipre,dim=-1).mean()
    kinetic_loss = torch.linalg.norm(trgt_kinetic- pred_kinetic,dim=-1).mean()

    momentum_loss = torch.linalg.norm(trgt_momentum- pred_momentum,dim=-1).mean()
    #print(MPJPE,pos_loss,kinetic_loss,momentum_loss)
    total_loss = (
            MPJPE+kinetic_loss+momentum_loss
    )

    return total_loss
