# data.py
import random
import torch
from torch.utils.data import Dataset


class RecommenderDataset(Dataset):
    """
    数据格式:
      {
          "seq": [item_id, item_id, ...],
          "next": 下一个 item_id,
          "length": 序列有效长度
      }
    """
    def __init__(self, df, max_seq_len, num_negative_samples, total_item_num):
        self.df = df.reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.num_negative_samples = num_negative_samples
        self.total_item_num = total_item_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["seq"][:self.max_seq_len]
        next_item = row["next"]
        length = row["len_seq"]

        # 生成唯一的负样本
        negatives = set()
        while len(negatives) < self.num_negative_samples:
            neg = random.randint(1, self.total_item_num)
            if neg != next_item:
                negatives.add(neg)
        negatives = list(negatives)

        return {"seq": seq, "next": next_item, "negatives": negatives, "length": length}


def collate_fn(batch):
    seqs = [item["seq"] for item in batch]
    next_items = [item["next"] for item in batch]
    negatives = [item["negatives"] for item in batch]
    lengths = [item["length"] for item in batch]
    return {"seqs": seqs, "next_items": next_items, "negatives": negatives, "lengths": lengths}