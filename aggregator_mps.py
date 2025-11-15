# 文件名: aggregator_mps.py
# -*- coding: utf-8 -*-
import torch
import logging
import mps_similarity as mps

class MPSAggregator:
    """
    MPS 聚合器：
    1. 接收各客户端的 Logits 和 Targets。
    2. 调用 mps_similarity 计算 Margin Profile。
    3. 计算相似度矩阵 S。
    4. 将相似度转换为聚合权重 w。
    """
    def __init__(self, device, gamma=1.0, temperature=0.1):
        self.device = device
        self.gamma = gamma
        self.temperature = temperature
        # 定义分位点，可以根据需要调整
        self.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)

    def compute_weights(self, logits_list, targets_list):
        """
        计算聚合权重
        :param logits_list: List[Tensor], 每个元素形状为 [N_samples, C]
        :param targets_list: List[Tensor], 每个元素形状为 [N_samples]
        """
        # 1. 检查数据有效性
        valid_logits = []
        valid_targets = []
        
        # 过滤掉空的数据（以防某个客户端验证集加载失败）
        for l, t in zip(logits_list, targets_list):
            if l is not None and t is not None and l.shape[0] > 0:
                valid_logits.append(l.to(self.device))
                valid_targets.append(t.to(self.device))
            else:
                logging.warning("发现无效或空的 logits/targets，跳过该客户端。")
                # 为了保持索引对应，插入全零（影响极小）或者处理索引映射。
                # 这里简单起见，我们假设所有客户端都成功，否则回退。
                return None 

        if len(valid_logits) < 2:
            logging.warning("有效客户端数量少于2，无法计算相似度。")
            return None

        try:
            # 2. 计算相似度矩阵 S [M, M]
            # 使用 mps_similarity.py 提供的端到端函数
            S_mps = mps.mps_from_logits_targets(
                logits_list=valid_logits,
                targets_list=valid_targets,
                quantiles=self.quantiles,
                gamma=self.gamma
            )
            
            logging.info(f"MPS 相似度矩阵:\n{S_mps.cpu().numpy()}")

            # 3. 将相似度转换为权重
            # w_i ∝ sum_j S_ij
            # 使用带温度的 softmax 使权重分布更平滑或更尖锐
            w_score = S_mps.sum(dim=1) # [M]
            
            if self.temperature > 0:
                weights = torch.softmax(w_score / self.temperature, dim=0)
            else:
                weights = w_score / w_score.sum()
                
            return weights

        except Exception as e:
            logging.error(f"MPS 计算过程中出错: {e}")
            return None