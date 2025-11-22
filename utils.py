import torch
import torch.nn.functional as F


# 前向 KL 散度： KL(teacher || student)
def compute_fkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0):

    # 1. 温度缩放
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    # 2. 得到概率与 log_probs
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()

    # 3. KL = sum P_t * (log P_t - log P_s)
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    # 4. 只对 label != padding_id 的位置计算
    mask = labels != padding_id
    kl = kl * mask

    # 5. 平均 KL，并乘温度²
    return kl.sum() / mask.sum() * (temp ** 2)


def compute_rkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0):
    # 温度缩放
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    # 计算 log_probs
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # 概率
    student_probs = student_log_probs.exp()

    # RKL = sum P_s * (log P_s - log P_t)
    rkl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    # 只在非 padding 位置计算
    mask = labels != padding_id
    rkl = rkl * mask

    # 温度²
    return rkl.sum() / mask.sum() * (temp ** 2)



# 偏向前向 KL
def compute_skewed_fkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0, skew_lambda=0.1):
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * student_probs
    mixed_log_probs = torch.log(mixed_probs + 1e-10)  # 避免 log(0)

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    kl = (teacher_probs * (teacher_log_probs - mixed_log_probs)).sum(dim=-1)

    mask = labels != padding_id
    kl = kl * mask

    return kl.sum() / mask.sum() * (temp ** 2)


# 偏向反向 KL
def compute_skewed_rkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0, skew_lambda=0.1):
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * student_probs
    mixed_log_probs = torch.log(mixed_probs + 1e-10)

    student_log_probs = F.log_softmax(student_logits, dim=-1)

    kl = (student_probs * (student_log_probs - mixed_log_probs)).sum(dim=-1)

    mask = labels != padding_id
    kl = kl * mask

    return kl.sum() / mask.sum() * (temp ** 2)
