import torch
import torch.nn.functional as F

# 计算熵
def calculate_entropy(prob_dist):
    """
    熵 H(P)
    """
    prob_dist = torch.clamp(prob_dist, min=1e-10)
    return -torch.sum(prob_dist * torch.log(prob_dist), dim=1)


def approximate_mutual_information(true_labels, pred_probs):
    """
    使用交叉熵近似互信息：
    I(C, C') ≈ H(C) - CrossEntropy(C, C')
    """
    true_dist = true_labels.float()  # 将标签从 [batch_size, height, width] 转为 [batch_size, 1, height, width]

    # 计算真实标签的熵 H(C)
    H_C = calculate_entropy(true_dist.unsqueeze(1))  # 真实标签的熵，取平均值

    # 计算交叉熵
    cross_entropy = F.binary_cross_entropy(pred_probs, true_dist, reduction="mean")  # 计算交叉熵

    # 计算互信息：I(C, C') ≈ H(C) - CrossEntropy(C, C')
    return H_C - cross_entropy

# 改进的 VI 损失函数
def vi_loss(true_labels, pred_logits, temperature=1.0):

    # 通过 sigmoid 获取预测概率
    pred_probs = torch.sigmoid(pred_logits / temperature)  # 使用温度参数平滑预测概率

    # 选择类别 1（晶界）的概率，形状为 [batch_size, height, width]
    pred_probs = pred_probs[:, 1, :, :]

    true_labels = true_labels.float()

    # 真实标签 H(C)
    true_dist = true_labels.unsqueeze(1)
    H_C = calculate_entropy(true_dist)

    # 计算预测标签的熵 H(C')
    H_C_prime = calculate_entropy(pred_probs.unsqueeze(1))

    # 计算互信息 I(C, C')，通过交叉熵近似
    I_C_C_prime = approximate_mutual_information(true_labels, pred_probs)

    # VI
    VI = H_C + H_C_prime - 2 * I_C_C_prime
    return VI.mean()
