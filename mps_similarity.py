# mps_similarity.py
# -*- coding: utf-8 -*-
"""
Margin Profile Similarity (MPS)

用“间隔分布轮廓”刻画客户端之间的相似度：
  1) 对每个客户端计算分类 margin = logit_y - max_{c!=y} logit_c
  2) 对 margin 取一组分位点，得到 margin 轮廓向量 M_i(q_1..q_K)
  3) 客户端 i,j 的距离 = 加权 L1(M_i, M_j)
  4) 相似度 S_ij = exp(-gamma * distance)

接口设计为：
  - 先对每个客户端调用 compute_margin_profile(...)
  - 再把所有 profile 传给 mps_similarity_matrix(...)
"""

from __future__ import annotations
from typing import Sequence, List, Tuple, Optional
import torch


@torch.no_grad()
def compute_margins_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    计算分类 margin:
      m(x) = logits[y] - max_{c!=y} logits[c]

    参数
    ----
    logits: [N, C]  或 [N, C, ...]，C 为类别数
    targets: [N]，长整型标签，取值范围 [0, C-1]

    返回
    ----
    margins: [N'] 1D tensor，每个样本一个 margin（越大越自信）
    """
    if logits.dim() > 2:
        # 展平成 [N', C]
        C = logits.shape[1]
        logits = logits.view(logits.size(0), C, -1).permute(0, 2, 1).reshape(-1, C)
        # 对应地展开 targets
        targets = targets.view(-1)
    else:
        # [N, C]
        pass

    # 取正确类别 logit
    # logits.gather(dim=1, index=targets[:, None]) -> [N,1]
    true_logits = logits.gather(dim=1, index=targets.long().view(-1, 1)).squeeze(1)

    # 取错误类别最大 logit
    # 方法：用一个很小的 mask 置掉正确类，再取 max
    C = logits.size(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(logits.size(0)), targets.long()] = False
    # 用 -inf 屏蔽正确类
    wrong_logits = logits.masked_fill(~mask, float("-inf"))
    max_wrong_logits, _ = wrong_logits.max(dim=1)

    margins = true_logits - max_wrong_logits  # [N]
    return margins


@torch.no_grad()
def margin_quantile_profile(
    margins: torch.Tensor,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> torch.Tensor:
    """
    从一堆 margin 样本中，计算分位点轮廓向量。

    参数
    ----
    margins: [N] 1D tensor
    quantiles: 分位点列表，例如 (0.1, 0.25, 0.5, 0.75, 0.9)

    返回
    ----
    profile: [K] 分位点向量，对应每个 q 的 margin 值
    """
    if margins.numel() == 0:
        # 极端情况：没有样本，用全零占位
        return torch.zeros(len(quantiles), dtype=torch.float32, device=margins.device)

    q = torch.tensor(quantiles, dtype=torch.float32, device=margins.device)
    # torch.quantile 支持 dim=None 直接对向量求分位数
    profile = torch.quantile(margins, q)
    return profile


@torch.no_grad()
def mps_distance(
    profile_i: torch.Tensor,
    profile_j: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    p: float = 1.0,
) -> torch.Tensor:
    """
    两个 margin 轮廓之间的距离，默认加权 L1。

    参数
    ----
    profile_i, profile_j: [K] 分位向量
    weight: [K] 或 None，对不同分位给予权重（例如高分位更重）
    p: 使用 L^p 范数，默认 1.0 (L1)

    返回
    ----
    d_ij: 标量 tensor
    """
    assert profile_i.shape == profile_j.shape, "margin profiles must have same shape"
    diff = torch.abs(profile_i - profile_j)

    if weight is not None:
        weight = weight.to(diff.device)
        diff = diff * weight

    if p == 1.0:
        d = diff.sum()
    else:
        d = (diff**p).sum().pow(1.0 / p)
    return d


@torch.no_grad()
def mps_similarity_matrix(
    profiles: Sequence[torch.Tensor],
    gamma: float = 1.0,
    weight: Optional[torch.Tensor] = None,
    p: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    根据一组客户端的 margin 轮廓，计算 MPS 相似度矩阵。

    参数
    ----
    profiles: 长度为 M 的列表，每个元素为 [K] 分位向量
    gamma: 相似度的温度系数： S = exp(-gamma * d)
    weight: [K] 或 None，用于加权不同分位（例如对高分位更敏感）
    p: 距离的 L^p 范数指数
    eps: 数值稳定用

    返回
    ----
    S: [M, M] 相似度矩阵，对称，主对角线接近 1
    """
    M = len(profiles)
    if M == 0:
        return torch.empty(0, 0)

    device = profiles[0].device
    S = torch.zeros(M, M, dtype=torch.float32, device=device)

    # 可选：预处理权重
    if weight is not None and not torch.is_tensor(weight):
        weight = torch.tensor(weight, dtype=torch.float32, device=device)

    for i in range(M):
        S[i, i] = 1.0
        for j in range(i + 1, M):
            d_ij = mps_distance(profiles[i], profiles[j], weight=weight, p=p)
            s_ij = torch.exp(-gamma * torch.clamp(d_ij, min=0.0))
            S[i, j] = s_ij
            S[j, i] = s_ij

    # 防止极端情况下全为 0
    S = torch.clamp(S, min=eps, max=1.0)
    return S


@torch.no_grad()
def compute_client_margin_profile(
    logits_list: Sequence[torch.Tensor],
    targets_list: Sequence[torch.Tensor],
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> List[torch.Tensor]:
    """
    一次性对多个客户端计算 margin 分位轮廓。

    参数
    ----
    logits_list: 长度为 M 的列表，第 i 个元素为该客户端的 logits [N_i, C] 或 [N_i, C, ...]
    targets_list: 长度为 M 的标签列表，第 i 个元素为 [N_i]

    返回
    ----
    profiles: 长度为 M 的列表，每个为 [K] 分位向量
    """
    assert len(logits_list) == len(targets_list), "logits_list and targets_list must match in length"

    profiles: List[torch.Tensor] = []
    for logits, targets in zip(logits_list, targets_list):
        margins = compute_margins_from_logits(logits, targets)
        prof = margin_quantile_profile(margins, quantiles=quantiles)
        profiles.append(prof)
    return profiles


# 可选：一个端到端辅助函数
@torch.no_grad()
def mps_from_logits_targets(
    logits_list: Sequence[torch.Tensor],
    targets_list: Sequence[torch.Tensor],
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    gamma: float = 1.0,
    weight: Optional[torch.Tensor] = None,
    p: float = 1.0,
) -> torch.Tensor:
    """
    端到端：从每个客户端的 logits/targets 直接得到 MPS 相似度矩阵。

    参数
    ----
    logits_list, targets_list: 每个客户端一对 [logits, labels]
    quantiles, gamma, weight, p: 同上

    返回
    ----
    S: [M, M] MPS 相似度矩阵
    """
    profiles = compute_client_margin_profile(
        logits_list=logits_list,
        targets_list=targets_list,
        quantiles=quantiles,
    )
    S = mps_similarity_matrix(
        profiles=profiles,
        gamma=gamma,
        weight=weight,
        p=p,
    )
    return S
