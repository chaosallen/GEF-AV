import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import scipy
import pandas as pd
import math
from skimage.morphology import skeletonize
from medpy.metric import binary
from skimage import morphology, graph
from random import randint
import logging

def soft_skel(x, iter_num=10):
    """
    Differentiable skeletonization
    x: probability map in [0,1], shape [B,1,H,W]
    """
    for _ in range(iter_num):
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(F.max_pool2d(min_pool, 3, 1, 1) - min_pool)
        x = F.relu(x - contour)
    return x

class CFLoss(nn.Module):
    """
    Connectivity-Focused Loss
    For artery/vein segmentation (2 channels, no background)
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        alpha: Dice weight
        beta: connectivity (skeleton) weight
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def dice_loss(self, pred, gt):
        """
        pred, gt: [B,1,H,W]
        """
        inter = (pred * gt).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + gt.sum(dim=(2,3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def connectivity_loss(self, pred, gt):
        """
        Skeleton consistency loss
        """
        skel_pred = soft_skel(pred)
        skel_gt = soft_skel(gt)

        inter = (skel_pred * skel_gt).sum(dim=(2,3))
        union = skel_pred.sum(dim=(2,3)) + skel_gt.sum(dim=(2,3))

        cl = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - cl.mean()

    def forward(self, pred_logits, gt):
        """
        pred_logits: [B,2,H,W]
        gt: [B,2,H,W]
        """
        pred = torch.sigmoid(pred_logits)

        loss = 0.0
        for c in range(2):  # artery & vein
            p = pred[:, c:c+1]
            g = gt[:, c:c+1]

            dice = self.dice_loss(p, g)
            conn = self.connectivity_loss(p, g)

            loss += self.alpha * dice + self.beta * conn

        return loss / 2



class clDiceLoss(nn.Module):
    """
    clDice loss for Artery-Vein Segmentation
    Channel 0: artery
    Channel 1: vein
    """
    def __init__(self, iter_num=10):
        super().__init__()
        self.iter_num = iter_num

    def forward(self, logits, targets):
        """
        logits : B,2,H,W  (raw network output)
        targets: B,2,H,W  (binary ground truth)
        """
        probs = torch.sigmoid(logits)

        cldice_a = cldice_single(
            probs[:, 0:1], targets[:, 0:1]
        )

        cldice_v = cldice_single(
            probs[:, 1:2], targets[:, 1:2]
        )

        # average artery & vein
        cldice = 0.5 * (cldice_a + cldice_v)

        return 1.0 - cldice



def cldice_single(pred, target, eps=1e-6):
    """
    pred, target: B,1,H,W  (probabilities in [0,1])
    """
    skel_pred = soft_skel(pred)
    skel_gt   = soft_skel(target)

    tprec = (skel_pred * target).sum(dim=(2,3)) / (skel_pred.sum(dim=(2,3)) + eps)
    tsens = (skel_gt * pred).sum(dim=(2,3)) / (skel_gt.sum(dim=(2,3)) + eps)

    cl_dice = (2 * tprec * tsens) / (tprec + tsens + eps)
    return cl_dice.mean()



def soft_erode(img):
    if img.ndimension() == 4:  # B,C,H,W
        p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=1, padding=(1,0))
        p2 = -F.max_pool2d(-img, kernel_size=(1,3), stride=1, padding=(0,1))
        return torch.min(p1, p2)
    else:
        raise ValueError("Unsupported tensor shape")


def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_num=10):
    """
    Soft skeletonization
    img: probability map in [0,1], shape B,C,H,W
    """
    skel = torch.zeros_like(img)
    img1 = img

    for _ in range(iter_num):
        opened = soft_open(img1)
        delta = F.relu(img1 - opened)
        skel = skel + delta
        img1 = soft_erode(img1)

    return skel



class CrossFromSegLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()  # 输入是 sigmoid 后概率

    def forward(self, pred_seg_logits, gt_cross):
        # pred_seg_logits: [B,2,H,W], logits
        # gt_cross: [B,1,H,W], 0/1

        pred_artery = torch.sigmoid(pred_seg_logits[:,0:1,:,:])
        pred_vein = torch.sigmoid(pred_seg_logits[:,1:2,:,:])
        pred_cross = pred_artery * pred_vein

        # BCE Loss
        bce_loss = self.bce(pred_cross, gt_cross)

        # Dice Loss
        smooth = 1e-6
        intersection = (pred_cross * gt_cross).sum(dim=(1,2,3))
        union = pred_cross.sum(dim=(1,2,3)) + gt_cross.sum(dim=(1,2,3))
        dice_loss = 1 - (2*intersection + smooth)/(union + smooth)
        dice_loss = dice_loss.mean()

        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss


class CrossBCEFromSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()  # 输入是 sigmoid 后概率

    def forward(self, pred_seg_logits, gt_cross):
        """
        pred_seg_logits: [B,2,H,W] 网络分割输出 logits
        gt_cross: [B,1,H,W] 交叉点标签 0/1
        """
        # 生成交叉点概率
        pred_artery = torch.sigmoid(pred_seg_logits[:,0:1,:,:])
        pred_vein   = torch.sigmoid(pred_seg_logits[:,1:2,:,:])
        pred_cross  = pred_artery * pred_vein  # [B,1,H,W]

        # 计算 BCE
        loss = self.bce(pred_cross, gt_cross)
        return loss

import numpy as np
from skimage import morphology, graph
from random import randint


def topo_metric(gt, pred, thresh, n_paths):
    """
    计算拓扑度量：INF（不可达路径）和COR（正确路径）

    参数：
        gt: 真实标签（二值图像，0或1）
        pred: 预测结果（二值图像，0或1）
        thresh: 阈值（用于二值化预测）
        n_paths: 随机采样的路径数量

    返回：
        (inf_count, mid_count, cor_count): 不可达、中间、正确路径的计数
    """
    result = []

    # 确保输入为二值 - 修复1：使用 int 而不是 np.int
    gt_bw = (gt > 0.5).astype(int)

    # 修复2：修正类型检查逻辑
    if isinstance(pred, np.ndarray):
        # 如果已经是整数或布尔类型，直接转换；否则根据阈值二值化
        if pred.dtype.kind in 'ui' or pred.dtype == bool:
            pred_bw = pred.astype(int)
        else:
            pred_bw = (pred > thresh).astype(int)
    else:
        pred_bw = (pred > thresh).astype(int)

    # 获取连通组件
    pred_cc = morphology.label(pred_bw)

    # 获取骨架线
    gt_cent = morphology.skeletonize(gt_bw)
    gt_cent_cc = morphology.label(gt_cent)
    pred_cent = morphology.skeletonize(pred_bw)

    # 代价矩阵（用于最短路径计算）
    gt_cost = np.ones(gt_cent.shape)
    gt_cost[gt_cent == 0] = 10000
    pred_cost = np.ones(pred_cent.shape)
    pred_cost[pred_cent == 0] = 10000

    # 获取骨架点坐标
    R_gt_cent, C_gt_cent = np.where(gt_cent == 1)

    # 如果没有足够的骨架点，返回空结果
    if len(R_gt_cent) < 2:
        return 0, 0, 0

    for _i in range(n_paths):
        # 随机选择第一个点 - 修复3：转换为 Python int
        idx1 = randint(0, len(R_gt_cent) - 1)
        label = gt_cent_cc[R_gt_cent[idx1], C_gt_cent[idx1]]
        ptx1 = (int(R_gt_cent[idx1]), int(C_gt_cent[idx1]))

        # 选择同一连通区域内的第二个点
        R_gt_cent_label, C_gt_cent_label = np.where(gt_cent_cc == label)
        if len(R_gt_cent_label) < 2:
            continue
        idx2 = randint(0, len(R_gt_cent_label) - 1)
        ptx2 = (int(R_gt_cent_label[idx2]), int(C_gt_cent_label[idx2]))

        # 检查预测中两点是否在同一连通区域
        if (pred_cc[ptx1] != pred_cc[ptx2]) or pred_cc[ptx1] == 0:
            result.append(0)  # 不可达
        else:
            # 找到预测骨架中的对应点
            R_pred_cent, C_pred_cent = np.where(pred_cent == 1)
            if len(R_pred_cent) == 0:
                result.append(0)
                continue

            poss_corr = np.column_stack((R_pred_cent, C_pred_cent))
            dist2_ptx1 = np.sum((poss_corr - np.array(ptx1)) ** 2, axis=1)
            dist2_ptx2 = np.sum((poss_corr - np.array(ptx2)) ** 2, axis=1)
            corr1 = tuple(poss_corr[np.argmin(dist2_ptx1)].astype(int))
            corr2 = tuple(poss_corr[np.argmin(dist2_ptx2)].astype(int))

            # 计算真实路径长度
            try:
                gt_path, _cost1 = graph.route_through_array(gt_cost, ptx1, ptx2)
                if gt_path is None or len(gt_path) < 2:
                    result.append(0)
                    continue
                gt_path = np.asarray(gt_path)
                path_gt_length = np.sum(np.sqrt(np.sum(np.diff(gt_path, axis=0) ** 2, axis=1)))
            except Exception:
                result.append(0)
                continue

            # 计算预测路径长度
            try:
                pred_path, _cost2 = graph.route_through_array(pred_cost, corr1, corr2)
                if pred_path is None:
                    result.append(0)
                    continue
                pred_path = np.asarray(pred_path)
                if pred_path.shape[0] < 2:
                    result.append(2)  # 单点视为正确
                else:
                    path_pred_length = np.sum(np.sqrt(np.sum(np.diff(pred_path, axis=0) ** 2, axis=1)))
                    if path_pred_length == 0:
                        result.append(2)
                    else:
                        ratio = path_gt_length / path_pred_length
                        if ratio < 0.9 or ratio > 1.1:
                            result.append(1)  # 长度差异大
                        else:
                            result.append(2)  # 正确
            except Exception:
                result.append(0)

    return result.count(0), result.count(1), result.count(2)

def compute_metrics(pred, gt, topo_thresh=0.5, topo_n_paths=500):
    """
    计算二值图像的分割性能指标：
    Dice, IoU, Accuracy, Sensitivity, Specificity, HD95, INF, COR
    参数：
        pred: 预测掩膜（ndarray，二值，0或1）
        gt:   真值掩膜（ndarray，二值，0或1）
        topo_thresh: 拓扑度量的阈值（默认0.5）
        topo_n_paths: 拓扑度量的随机路径数（默认100）
    返回：
        metrics: dict，包含每个指标名称及其值
    """
    from skimage import morphology, graph
    from random import randint

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()

    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    acc = (TP + TN) / (TP + FP + TN + FN + 1e-8)
    se = TP / (TP + FN + 1e-8)  # Sensitivity
    sp = TN / (TN + FP + 1e-8)  # Specificity
    cl_dice = compute_clDice(pred, gt)

    # HD95 计算，若预测或标签全为0，设为nan
    if np.any(pred) and np.any(gt):
        try:
            hd95 = binary.hd95(pred, gt)
        except:
            hd95 = np.nan
    else:
        hd95 = np.nan

    # 计算拓扑度量 INF 和 COR
    try:
        # 确保输入为二值图像且类型正确
        gt_binary = gt.astype(np.uint8)
        pred_binary = pred.astype(np.uint8)

        # 调用拓扑度量函数
        inf_count, _, cor_count = topo_metric(gt_binary, pred_binary, topo_thresh, topo_n_paths)

        # 转换为比率（0-1之间），便于与其他指标比较
        total_paths = inf_count + cor_count + (topo_n_paths - inf_count - cor_count)
        if total_paths > 0:
            inf = inf_count / total_paths  # 不可达路径比例
            cor = cor_count / total_paths  # 正确路径比例
        else:
            inf = np.nan
            cor = np.nan
    except Exception as e:
        # 如果拓扑度量计算失败，设为nan
        inf = np.nan
        cor = np.nan
        logging.warning(f"拓扑度量计算失败: {e}")

    return {
        'Dice': dice,
        'IoU': iou,
        'Accuracy': acc,
        'Sensitivity': se,
        'Specificity': sp,
        'clDice': cl_dice,
        'HD95': hd95,
        'INF': inf,  # 不可达路径比例
        'COR': cor  # 正确路径比例
    }


def compute_clDice(pred, gt):
    """
    计算clDice指标
    pred, gt: 二值数组, shape (H, W)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    # 提取骨架
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)

    # Precision-like: 预测骨架落在GT区域的比例
    P = (skel_pred & gt).sum() / (skel_pred.sum() + 1e-8)

    # Recall-like: GT骨架落在预测区域的比例
    R = (skel_gt & pred).sum() / (skel_gt.sum() + 1e-8)

    cl_dice = 2 * P * R / (P + R + 1e-8)
    return cl_dice


def evaluate_and_save(preds, gts, names, save_csv_path):
    """
    对多个样本进行评估，并保存为 CSV 文件。
    preds: List of ndarray, 每个元素形状为 [2, H, W]，二通道预测掩膜（0/1）
    gts: List of ndarray, 每个元素形状为 [2, H, W]，二通道真值掩膜（0/1）
    names: List[str]，每个样本的名称
    """
    results = []
    for i in range(len(preds)):
        pred = preds[i]
        gt = gts[i]
        result = {'name': names[i]}
        for c, label in enumerate(['artery', 'vein']):
            pred_c = pred[c]
            gt_c = gt[c]
            metrics = compute_metrics(pred_c, gt_c)
            for key, value in metrics.items():
                result[f'{label}_{key}'] = value
        results.append(result)

    # 构建DataFrame
    df = pd.DataFrame(results)

    # === Step 1: 计算所有样本的均值行 ===
    mean_row = df.iloc[:, 1:].mean().to_frame().T
    mean_row.insert(0, 'name', 'Mean')

    # === Step 2: 基于 mean_row 计算 artery+vein 平均 ===
    avg_metrics = {}
    for metric in ['Dice', 'IoU', 'Accuracy', 'Sensitivity', 'clDice', 'HD95']:
        artery_col = f'artery_{metric}'
        vein_col = f'vein_{metric}'
        if artery_col in mean_row.columns and vein_col in mean_row.columns:
            avg_metrics[f'avg_{metric}'] = (mean_row[artery_col].values[0] + mean_row[vein_col].values[0]) / 2

    # 把 avg_metrics 加入 mean_row
    for k, v in avg_metrics.items():
        mean_row[k] = v

    # 拼接回 df
    df = pd.concat([df, mean_row], ignore_index=True)

    # 保存
    df.to_csv(save_csv_path, index=False)
    print(f"Saved evaluation results to {save_csv_path}")

    # === 终端打印最终的 artery+vein 平均 ===
    print("\n=== Final Artery+Vein Average Metrics (all samples) ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

def feature_kl_distillation(feats_s, feats_t, temperature=1.0, weight_dict=None):
    """
    使用 KL 散度评估学生和教师网络之间的特征分布差异
    - feats_s, feats_t: dict，{conv2: tensor, conv3: tensor, conv4: tensor}
    - temperature: 蒸馏温度
    - weight_dict: 各层加权系数
    """
    loss = 0.0
    eps = 1e-6

    for name in feats_s.keys():
        fs = feats_s[name]
        ft = feats_t[name]

        # 特征展平为 [B, C*H*W]
        B, C, H, W = fs.shape
        fs_flat = fs.view(B, -1) / temperature
        ft_flat = ft.view(B, -1) / temperature

        # Softmax 转换为概率分布
        fs_prob = F.log_softmax(fs_flat, dim=1)  # student: log(P)
        ft_prob = F.softmax(ft_flat, dim=1)      # teacher: Q

        kl = F.kl_div(fs_prob, ft_prob, reduction='batchmean', log_target=False)

        if weight_dict is not None:
            kl *= weight_dict.get(name, 1.0)

        loss += kl

    loss /= len(feats_s)
    return loss



def gaussian_kernel_2d(kernel_size=25, sigma=25, device='cpu'):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g1d = torch.exp(- (coords ** 2) / (2 * sigma ** 2))
    g1d /= g1d.sum()
    g2d = torch.outer(g1d, g1d)
    kernel = g2d.unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,k,k)
    return kernel

def segment_to_energy_field(pred_artery, pred_vein, kernel):
    """
    输入:
        pred_artery, pred_vein: [B,1,H,W], 预测的动脉和静脉概率图（sigmoid后）
        kernel: 高斯核，shape (1,1,k,k)
    输出:
        energy_m: [B,1,H,W], 先对单通道动静脉做高斯卷积后相减，得到混合能量场
    """
    artery_blur = F.conv2d(pred_artery, kernel, padding=kernel.shape[-1]//2)
    vein_blur = F.conv2d(pred_vein, kernel, padding=kernel.shape[-1]//2)
    energy_m = artery_blur - vein_blur
    return energy_m

def segment_to_energy_field_v2(pred_artery, pred_vein, sigma=25):
    """
    使用与标签一致的方式生成模拟能量场：
    对动脉/静脉概率图 sigmoid 后做二值化 -> 取反 -> distance transform -> exp(-d^2/2σ²)
    然后做差（动脉能量 - 静脉能量）生成 energy map。

    输入:
        pred_artery, pred_vein: [B,1,H,W]，sigmoid后 or 概率图
        sigma: 控制衰减速率的高斯参数
    输出:
        energy_m: [B,1,H,W]
    """
    def _to_energy(tensor):
        np_tensor = tensor.detach().cpu().numpy()
        energy_list = []
        for i in range(np_tensor.shape[0]):
            binary = (np_tensor[i, 0] > 0.5).astype(np.uint8)
            inverted = 1 - binary
            dist = scipy.ndimage.distance_transform_edt(inverted)
            energy = np.exp(- (dist ** 2) / (2 * sigma ** 2))
            energy_list.append(energy)
        energy_array = np.stack(energy_list, axis=0)  # shape [B, H, W]
        return torch.from_numpy(energy_array).unsqueeze(1).to(tensor.device).float()

    energy_a = _to_energy(pred_artery)
    energy_v = _to_energy(pred_vein)
    energy_m = energy_a - energy_v
    return energy_m


def getConfusionMatrixInformation(t_labels, p_labels):
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(t_labels, p_labels)
    # print(f"混淆矩阵：\n{conf_matrix}")
    # 计算 TP, TN, FP, FN
    num_classes = len(np.unique(t_labels))
    # 获取对角元素
    TP = np.diag(conf_matrix)

    # 纵向FP
    FP = np.sum(conf_matrix, axis=0) - TP
    # 横向FN
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = []
    for i in range(num_classes):
        temp = np.delete(conf_matrix, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(np.sum(temp))
    return TP,TN,FP,FN


def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir,name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir','\''+dir+'\'','is created.')

def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)


def cal_acc(img1,img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] == img2[i,j]:
                acc += 1
    return acc/(shape[0]*shape[1])

def cal_miou(img1,img2):
    classnum = img2.max()
    iou=np.zeros((int(classnum),1))
    for i in range(int(classnum)):
        imga=img1==i+1
        imgb=img2==i+1
        imgi=imga * imgb
        imgu=imga + imgb
        iou[i]=np.sum(imgi)/np.sum(imgu)
    miou=np.mean(iou)
    return miou


def cal_miou_multilabel(pred, label):
    """
    Args:
        pred: torch.Tensor or np.ndarray of shape [B, C, H, W] - predicted binary masks (0 or 1)
        label: torch.Tensor or np.ndarray of shape [B, C, H, W] - ground truth masks (0 or 1)
    Returns:
        miou: mean IoU over all channels
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    assert pred.shape == label.shape, f"Shape mismatch: pred {pred.shape}, label {label.shape}"

    class_num = pred.shape[1]
    ious = []

    for i in range(class_num):
        pred_i = pred[:, i, :, :]
        label_i = label[:, i, :, :]

        intersection = np.logical_and(pred_i, label_i).sum()
        union = np.logical_or(pred_i, label_i).sum()

        iou = intersection / union if union != 0 else 1.0  # 若无正样本视为完全重合
        ious.append(iou)

    miou = np.mean(ious)
    return miou

def make_one_hot(input, shape):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    result = torch.zeros(shape)
    result.scatter_(1, input.cpu(), 1)
    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        shape=predict.shape
        target = torch.unsqueeze(target, 1)
        target=make_one_hot(target.long(),shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]