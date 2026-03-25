import torch
import torch.nn.functional as F


# Forward KL Divergence: KL(teacher || student)
def compute_fkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0):
    """
    Compute Forward Kullback-Leibler (KL) Divergence between teacher and student distributions.

    Forward KL: KL(teacher || student) = sum teacher_probs * log(teacher_probs / student_probs)
    This formulation encourages the student to cover all modes of the teacher distribution.

    Args:
        student_logits: Logits from student model (batch, seq_len, vocab_size)
        teacher_logits: Logits from teacher model (batch, seq_len, vocab_size)
        labels: Ground truth labels for masking padding tokens (batch, seq_len)
        padding_id: Token ID used for padding (default: -100)
        temp: Temperature parameter for softening distributions (default: 1.0)

    Returns:
        Scalar tensor representing the average forward KL divergence
    """
    # 1. Apply temperature scaling to soften distributions
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    # 2. Compute log probabilities and probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()  # Convert log probs to probabilities

    # 3. Calculate KL divergence: sum P_t * (log P_t - log P_s)
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    # 4. Mask out padding positions (where labels == padding_id)
    mask = labels != padding_id
    kl = kl * mask

    # 5. Return average KL scaled by temperature squared
    # Temperature scaling factor: multiplying by temp^2 accounts for the temperature scaling applied earlier
    return kl.sum() / mask.sum() * (temp ** 2)


def compute_rkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0):
    """
    Compute Reverse Kullback-Leibler (RKL) Divergence between student and teacher distributions.

    Reverse KL: KL(student || teacher) = sum student_probs * log(student_probs / teacher_probs)
    This formulation encourages the student to focus on a single mode of the teacher distribution,
    often resulting in sharper, more confident predictions.

    Args:
        student_logits: Logits from student model (batch, seq_len, vocab_size)
        teacher_logits: Logits from teacher model (batch, seq_len, vocab_size)
        labels: Ground truth labels for masking padding tokens (batch, seq_len)
        padding_id: Token ID used for padding (default: -100)
        temp: Temperature parameter for softening distributions (default: 1.0)

    Returns:
        Scalar tensor representing the average reverse KL divergence
    """
    # Apply temperature scaling
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    # Compute log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Convert student log probs to probabilities
    student_probs = student_log_probs.exp()

    # Calculate reverse KL: sum P_s * (log P_s - log P_t)
    rkl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    # Mask out padding positions
    mask = labels != padding_id
    rkl = rkl * mask

    # Return average scaled by temperature squared
    return rkl.sum() / mask.sum() * (temp ** 2)


# Skewed Forward KL Divergence
def compute_skewed_fkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0, skew_lambda=0.1):
    """
    Compute Skewed Forward KL Divergence with a mixture of teacher and student distributions.

    This formulation uses a mixed distribution: (skew_lambda * teacher + (1-skew_lambda) * student)
    as the target, creating a smoother interpolation between forward and reverse KL behaviors.

    Args:
        student_logits: Logits from student model (batch, seq_len, vocab_size)
        teacher_logits: Logits from teacher model (batch, seq_len, vocab_size)
        labels: Ground truth labels for masking padding tokens (batch, seq_len)
        padding_id: Token ID used for padding (default: -100)
        temp: Temperature parameter for softening distributions (default: 1.0)
        skew_lambda: Skewing parameter controlling mixture weight (default: 0.1)
                    Higher values give more weight to teacher distribution

    Returns:
        Scalar tensor representing the average skewed forward KL divergence
    """
    # Apply temperature scaling
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    # Compute probabilities for both distributions
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Create mixed distribution: weighted combination of teacher and student
    mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * student_probs
    # Add small epsilon to avoid log(0)
    mixed_log_probs = torch.log(mixed_probs + 1e-10)

    # Teacher log probabilities for KL calculation
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Calculate skewed forward KL
    kl = (teacher_probs * (teacher_log_probs - mixed_log_probs)).sum(dim=-1)

    # Mask out padding positions
    mask = labels != padding_id
    kl = kl * mask

    # Return average scaled by temperature squared
    return kl.sum() / mask.sum() * (temp ** 2)


# Skewed Reverse KL Divergence
def compute_skewed_rkl(student_logits, teacher_logits, labels, padding_id=-100, temp=1.0, skew_lambda=0.1):
    """
    Compute Skewed Reverse KL Divergence with a mixture of teacher and student distributions.

    This formulation uses a mixed distribution: ((1-skew_lambda) * teacher + skew_lambda * student)
    as the target for reverse KL, creating an interpolation that balances mode-seeking
    and mode-covering behaviors.

    Args:
        student_logits: Logits from student model (batch, seq_len, vocab_size)
        teacher_logits: Logits from teacher model (batch, seq_len, vocab_size)
        labels: Ground truth labels for masking padding tokens (batch, seq_len)
        padding_id: Token ID used for padding (default: -100)
        temp: Temperature parameter for softening distributions (default: 1.0)
        skew_lambda: Skewing parameter controlling mixture weight (default: 0.1)
                    Note: Here teacher weight is (1-skew_lambda), student weight is skew_lambda

    Returns:
        Scalar tensor representing the average skewed reverse KL divergence
    """
    # Apply temperature scaling
    student_logits = student_logits / temp
    teacher_logits = teacher_logits / temp

    # Compute probabilities for both distributions
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Create mixed distribution: complementary weighting compared to skewed forward KL
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * student_probs
    # Add small epsilon to avoid log(0)
    mixed_log_probs = torch.log(mixed_probs + 1e-10)

    # Student log probabilities for reverse KL calculation
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # Calculate skewed reverse KL
    kl = (student_probs * (student_log_probs - mixed_log_probs)).sum(dim=-1)

    # Mask out padding positions
    mask = labels != padding_id
    kl = kl * mask

    # Return average scaled by temperature squared
    return kl.sum() / mask.sum() * (temp ** 2)