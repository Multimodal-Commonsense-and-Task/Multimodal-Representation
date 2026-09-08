import torch
import torch.nn.functional as F


################################################################################
# Table 1: Baseline Knowledge Distillation Loss Functions
################################################################################

def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss

def jeffrey(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill((teacher_probs - student_probs) * student_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill((teacher_probs - student_probs) * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def ab_div(logits, teacher_logits, no_model_batch, alpha=0.2, beta=0.7):
    """Calculate D^{(alpha, beta)} divergence."""
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    eps = 1e-8

    if abs(alpha) < eps and abs(beta) < eps:
        divergence = 0.5 * torch.sum((log_q - log_p).pow(2), dim=-1)

    elif abs(alpha) < eps:
        safe_log_ratio_div_beta = torch.where(
            torch.isfinite(log_q - log_p), log_q - log_p, 0.0
        )
        divergence = torch.sum(
            torch.exp(beta * log_q) * (beta * safe_log_ratio_div_beta - 1) + torch.exp(beta * log_p),
            dim=-1
        ) / (beta ** 2)

    elif abs(beta) < eps:
        safe_log_ratio_div_alpha = torch.where(
            torch.isfinite(log_p - log_q), log_p - log_q, 0.0
        )
        divergence = torch.sum(
            torch.exp(alpha * log_p) * (alpha * safe_log_ratio_div_alpha - 1) + torch.exp(alpha * log_q),
            dim=-1
        ) / (alpha ** 2)

    elif abs(alpha + beta) < eps:
        safe_log_r = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(
            alpha * safe_log_r + torch.exp(-alpha * safe_log_r) - 1,
            dim=-1
        ) / (alpha ** 2)

    else:
        apb = alpha + beta
        term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
        term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
        term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
        divergence = - (term1 - term2 - term3) / (alpha * beta)

    mask = (no_model_batch["label"] != -100).float()
    safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
    masked_sum = (safe_divergence * mask).sum()
    mask_sum = mask.sum()
    distil_loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum

    return distil_loss

# ============================================================
# Table 1: Main CSD Loss
# ============================================================

def csd(logits, teacher_logits, no_model_batch, temp=1.0):
    student_probs = F.softmax(logits / temp, dim=-1, dtype=torch.float32)
    loss = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits), dim=-1, keepdim=True)).detach() * student_probs.detach() * logits
    x = torch.sum(loss, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss

# ============================================================
# Figure 3b: CSD (U,S) and CSD (T,S)
# ============================================================

def csd_weighting_function(logits, teacher_logits, no_model_batch, type="US", epoch= None, total_epoch = None):
 
    if type == "US": ## CSD (U,S) Grad / p1: uniform / p2: student
        B, T, V = teacher_logits.shape
        uni_probs = torch.full((B, T, V), 1.0 / V, device=teacher_logits.device, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        loss1 = (logits - teacher_logits - torch.sum(uni_probs * (logits - teacher_logits), dim=-1,keepdim=True)).detach() * student_probs.detach() * logits
        loss2 = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits), dim=-1, keepdim=True)).detach() * uni_probs.detach() * logits
        loss = (loss1 + loss2) / 2

    elif type == "TS": ## CSD (T,S) Grad / p1: teacher / p2: student
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        loss1 = (logits - teacher_logits - torch.sum(teacher_probs * (logits - teacher_logits), dim=-1,keepdim=True)).detach() * student_probs.detach() * logits
        loss2 = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits), dim=-1,keepdim=True)).detach() * teacher_probs.detach() * logits
        loss = (loss1 + loss2) / 2

    x = torch.sum(loss, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

# ============================================================
# Table 5: DLD Loss for Weighting and Normalization Ablations
# ============================================================

def dld_ablation(logits, teacher_logits, no_model_batch, normalize="none", w="none", temp=1.0):
    if w == "S":
        weight = F.softmax(logits / temp, dim=-1, dtype=torch.float32).detach()
    elif w == "T":
        weight = F.softmax(teacher_logits / temp, dim=-1, dtype=torch.float32).detach()
    elif w == "U":
        B, T, V = teacher_logits.shape
        weight = torch.full((B, T, V), 1.0 / V, device=teacher_logits.device, dtype=torch.float32).detach()
    else:
        assert 0

    if normalize == "mean":
        logits = logits - logits.mean(axis=-1, keepdims=True)
        teacher_logits = teacher_logits - teacher_logits.mean(axis=-1, keepdims=True)
    elif normalize == "min":
        values, _ = logits.min(dim=-1, keepdim=True)
        logits -= values

        values, _ = teacher_logits.min(dim=-1, keepdim=True)
        teacher_logits -= values

    elif normalize == "max":
        values, _ = logits.max(dim=-1, keepdim=True)
        logits -= values
        values, _ = teacher_logits.max(dim=-1, keepdim=True)
        teacher_logits -= values
    elif normalize == "std":
        logits = (logits - logits.mean(axis=-1, keepdims=True)) / (logits.std(axis=-1, keepdims=True) + 1e-9)
        teacher_logits = (teacher_logits - teacher_logits.mean(axis=-1, keepdims=True)) / (teacher_logits.std(axis=-1, keepdims=True) + 1e-9)
    elif normalize == "range":
        x_min, _ = logits.min(axis=-1, keepdims=True)
        x_max, _ = logits.max(axis=-1, keepdims=True)
        logits = 2 * (logits - x_min) / (x_max - x_min + 1e-9) - 1
        t_min, _ = teacher_logits.min(axis=-1, keepdims=True)
        t_max, _ = teacher_logits.max(axis=-1, keepdims=True)
        teacher_logits = 2 * (teacher_logits - t_min) / (t_max - t_min + 1e-9) - 1
    elif normalize == "no":
        logits = logits
        teacher_logits = teacher_logits
    else:
        assert 0


    loss = weight.detach() * (logits - teacher_logits) ** 2 / 4
    x = torch.sum(loss, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

