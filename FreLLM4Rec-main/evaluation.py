# evaluation.py
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
def evaluate(model, data_loader, device, ks=[1, 5, 10, 15, 20]):
    model.eval()
    metrics_sum = {k: {'ndcg': 0.0, 'hit': 0.0, 'mrr': 0.0, 'count': 0} for k in ks}
    with torch.no_grad():
        # Add tqdm for the data loader loop
        for batch in tqdm(data_loader, desc="Evaluating batches"):
            logits_list = model(batch["seqs"], batch["next_items"], batch["negatives"])
            for logits in logits_list:
                sorted_scores, sorted_indices = torch.sort(logits, descending=True)
                rank = sorted_indices.tolist().index(0)
                for k in ks:
                    hit = 1.0 if rank < k else 0.0
                    ndcg = 1.0 / math.log2(rank + 2) if rank < k else 0.0
                    mrr = 1.0 / (rank + 1) if rank < k else 0.0
                    metrics_sum[k]['hit'] += hit
                    metrics_sum[k]['ndcg'] += ndcg
                    metrics_sum[k]['mrr'] += mrr
                    metrics_sum[k]['count'] += 1
    avg_metrics = {}
    for k in ks:
        cnt = metrics_sum[k]['count']
        avg_metrics[k] = (metrics_sum[k]['ndcg'] / (cnt + 1e-9),
                          metrics_sum[k]['hit'] / (cnt + 1e-9),
                          metrics_sum[k]['mrr'] / (cnt + 1e-9))
    return avg_metrics


