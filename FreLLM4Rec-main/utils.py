# utils.py
import os
import math
import random
import argparse
import torch


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinearWarmupCosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_steps, init_lr, min_lr, warmup_steps, warmup_start_lr, last_epoch=-1):
        self.max_steps = max_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.warmup_start_lr + (self.init_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_steps
                for _ in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (self.init_lr - self.min_lr) * cosine_decay
                for _ in self.base_lrs
            ]




def parse_args():
    parser = argparse.ArgumentParser(description="Recommender System with Llama2")
    # 运行模式
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                        help="运行模式：train 或 test")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="测试模式下要加载的checkpoint路径")
    
    # 数据路径
    parser.add_argument("--train_path", type=str, 
                        help="训练集 pickle 文件路径")
    parser.add_argument("--val_path", type=str, 
                        help="验证集 pickle 文件路径")
    parser.add_argument("--test_path", type=str, default="/data/mhwang/LLM/Fre_LLM/dataset/process_data/All_Beauty/Test_data.df",
                        help="测试集 pickle 文件路径")
    parser.add_argument("--id2name_path", type=str, default="/data/mhwang/LLM/Fre_LLM/dataset/process_data/All_Beauty/id2name.txt",
                        help="item id 到 title 的映射文件，格式：id::title")
    
    # 模型参数
    parser.add_argument("--pretrained_model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="预训练 Llama2 模型或路径")
    parser.add_argument("--max_seq_len", type=int, default=100,
                        help="序列最大长度")
    
    # 训练参数
    parser.add_argument("--num_negative_samples", type=int, default=5,
                        help="负样本采样数量")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="初始学习率")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine"],
                        help="学习率调度器类型")
    parser.add_argument("--lr_decay_min_lr", type=float, default=5e-6,
                        help="学习率调度器的最小学习率")
    parser.add_argument("--lr_warmup_start_lr", type=float, default=5e-6,
                        help="预热初始学习率")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="分布式训练时的 local_rank")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="权重衰减")
    parser.add_argument("--rec_model_path", type=str,
                        default="/data/mhwang/LLM/Fre_LLM/Pre_Train_Rec_Model/sasrec/All_Beauty/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth",
                        help="预训练 SASRec 模型路径")
    parser.add_argument("--proj_intermediate_dim", type=int, default=512,
                        help="Item embedding projection 模块中间层维度")
    
    parser.add_argument("--embed_mode", type=str, default="None",
                        choices=[None],
                        help="")
    parser.add_argument("--num_trainable_items", type=int, default=1,
                        help="可训练 item 数量 (使用 trainable 模态时启用，0 表示不使用)")
    
    parser.add_argument("--hidden_layer", type=int, default=-1,
                        help="用LLM的第几层")
    parser.add_argument("--inject_every_layer", action="store_true", default=False, help="是否在每一层注入ID")
    parser.add_argument("--inject_layer", type=int, default=0, help="输入id的layer")
    


    parser.add_argument('--graph_conv_weight', type=float, default=0.7, help='Weight of graph_conv_weight')
    parser.add_argument('--num_check', type=int, default=9, help='checkpoint num')



    
    parser.add_argument('--init_fa', type=float, default=0.1, help='Weight of fft')
    parser.add_argument('--lower_quantile', type=float, default=0.00, help='lower fre')
    parser.add_argument('--upper_quantile', type=float, default=0.25, help='upper fre')
    parser.add_argument("--new_item_emb_path", type=str,
                        default="/data/mhwang/LLM/Fre_LLM/Pre_Train_Rec_Model/LightGCN/LastFM/new_embedding.pt",
                        help="newc embedding path")
    
    args = parser.parse_args()
    
    print("\nParsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args

