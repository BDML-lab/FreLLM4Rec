
# main.py
import os
import math
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

from utils import parse_args, seed_everything, LinearWarmupCosineLRScheduler
from data import RecommenderDataset, collate_fn
from evaluation import evaluate
from transformers import AutoTokenizer
from GraphBuilder import ItemKNNGraphBuilder


from models.llama_for_rec import LlamaForRec

def extract_and_save_text_embeddings(model, tokenizer, id2name, total_item_num, device, output_dir, batch_size=128):
    text_emb_path = os.path.join(output_dir, "item_text_embeddings.pt")

    if os.path.exists(text_emb_path):
        print(f"[Info] 发现已存在的 text embeddings 文件: {text_emb_path}")
        text_embeddings = torch.load(text_emb_path, map_location=device)
        print(f"[Info] 已加载 text embeddings (shape: {text_embeddings.shape})")
        
        if text_embeddings.shape[0] != total_item_num:
            print(f"[Warning] 加载的 embeddings 大小 ({text_embeddings.shape[0]}) 与 total_item_num ({total_item_num}) 不匹配，将重新生成")
        else:
            return text_embeddings

    item_titles = [id2name.get(i, str(i)) for i in range(1, total_item_num + 1)]

    text_embeddings_list = []

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, total_item_num, batch_size):
            end_idx = min(start_idx + batch_size, total_item_num)
            batch_titles = item_titles[start_idx:end_idx]
            print(f"[Info] 处理批次: {start_idx} - {end_idx} / {total_item_num}")

            encoded_inputs = tokenizer(batch_titles, padding=True, truncation=True, return_tensors="pt", max_length=128)
            input_ids = encoded_inputs["input_ids"].to(device)          # [batch_size, seq_len]
            attention_mask = encoded_inputs["attention_mask"].to(device)  # [batch_size, seq_len]

            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

            batch_embeddings = last_hidden_states[:, -1, :]  # [batch_size, hidden_size]
            text_embeddings_list.append(batch_embeddings.cpu())  # 移动到 CPU 释放显存

            del input_ids, attention_mask, outputs, last_hidden_states, batch_embeddings
            torch.cuda.empty_cache()  # 清除 GPU 缓存

    # 将所有批次的 embeddings 拼接成完整张量
    text_embeddings = torch.cat(text_embeddings_list, dim=0)  # [total_item_num, hidden_size]
    print(f"[Info] 所有批次处理完成，合并后的 embeddings 形状: {text_embeddings.shape}")

    torch.save(text_embeddings, text_emb_path)  # 直接保存 CPU 张量
    print(f"[Info] Text embeddings 已保存到 {text_emb_path} (shape: {text_embeddings.shape})")

    return text_embeddings.to(device)

def main_worker(local_rank, args):
    print("Starting training/testing...")
    seed_everything(args.seed)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    args.device = device
    if args.mode == "train":
        df_train = pd.read_pickle(args.train_path)
        df_val = pd.read_pickle(args.val_path)
        df_test = pd.read_pickle(args.test_path)
    else:
        df_train = pd.read_pickle(args.train_path)
        df_val = pd.read_pickle(args.val_path)
        df_test = pd.read_pickle(args.test_path)
    
    # Load id -> title mapping with tqdm
    id2name = {}
    with open(args.id2name_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading id2name mapping"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) >= 2:
                try:
                    item_id = int(parts[0])
                    title = "::".join(parts[1:])
                    id2name[item_id] = title
                except Exception:
                    pass
    args.id2name = id2name  # Pass mapping to hparams
    
    # Determine upper bound for negative sampling
    total_item_num = max(
        df_train["seq"].map(max).max(),
        df_train["next"].max(),
        df_val["seq"].map(max).max(),
        df_val["next"].max(),
        df_test["seq"].map(max).max(),
        df_test["next"].max()
    )
    args.item_num = total_item_num
    print(f"[Info] Computed item_num: {args.item_num}")
    args.user_num = max(df_train.shape[0], df_val.shape[0], df_test.shape[0])
    print(f"[Info] Computed user_num: {args.user_num}")


    if args.mode == "train":
        train_dataset = RecommenderDataset(df_train, args.max_seq_len, args.num_negative_samples, total_item_num)
        val_dataset = RecommenderDataset(df_val, args.max_seq_len, args.num_negative_samples, total_item_num)
        test_dataset = RecommenderDataset(df_test, args.max_seq_len, args.num_negative_samples, total_item_num)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        test_dataset = RecommenderDataset(df_test, args.max_seq_len, args.num_negative_samples, total_item_num)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Load tokenizer and pass to hparams
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    args.tokenizer = tokenizer

    model = LlamaForRec(args)
    model.to(device)

    text_embeddings = extract_and_save_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        id2name=id2name,
        total_item_num=total_item_num,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size  
    )
    args.text_embeddings = text_embeddings 
    model.text_embeddings = text_embeddings
    
    

    
    print("[Info] Constructing UI matrix for Graph Fourier Transform...")
    user_interactions = [list(filter(lambda x: x != 0, seq)) for seq in df_test["seq"]]
    N_users = len(user_interactions)
    unique_items = list(range(1, total_item_num + 1))
    
    if 0 in unique_items:
        unique_items.discard(0)
    all_item_ids = sorted(list(unique_items))
    os.makedirs(args.output_dir + '/', exist_ok=True)
    
    knn_builder = ItemKNNGraphBuilder(
        model=model,
        item_ids=all_item_ids,
        output_dir=args.output_dir,
        device=device,
        topk=10,  # 可通过 args 指定
        norm_type='sym'  # 可通过 args 指定
    )

    knn_graph = knn_builder.build_knn_graph(force_rebuild=False)  # 默认使用缓存
    knn_graph_id = knn_builder.build_knn_graph_id(force_rebuild=False, df_test=df_test, item_num=total_item_num)  # 默认使用缓存

    print(f"[Info] KNN_ID graph shape: {knn_graph_id.shape}, sparsity: {knn_graph_id._nnz() / (knn_graph_id.shape[0] * knn_graph_id.shape[1]):.6f}")
    model.knn_graph_id = knn_graph_id
    
    print(f"[Info] KNN graph shape: {knn_graph.shape}, sparsity: {knn_graph._nnz() / (knn_graph.shape[0] * knn_graph.shape[1]):.6f}")
    model.knn_graph = knn_graph

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    

    
    if args.mode == "test":
        if not args.checkpoint or not os.path.isfile(args.checkpoint):
            raise ValueError("Test mode requires a valid --checkpoint")
        print(f"[Test] Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = model.state_dict()
        state_dict.update(checkpoint)
        model.load_state_dict(state_dict)
        return
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulate_grad_batches)
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = max_steps // 20
    scheduler = LinearWarmupCosineLRScheduler(optimizer, max_steps=max_steps, init_lr=args.lr,
                                              min_lr=args.lr_decay_min_lr,
                                              warmup_steps=warmup_steps,
                                              warmup_start_lr=args.lr_warmup_start_lr)
    
    train_losses = []
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        step_count = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} batches")):
            loss = model(batch["seqs"], batch["next_items"], batch["negatives"])
            loss = loss / args.accumulate_grad_batches
            loss.backward()
            epoch_loss += loss.item() * args.accumulate_grad_batches
            if ((i + 1) % args.accumulate_grad_batches == 0) or (i + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                global_step += 1
        avg_loss = epoch_loss / (step_count + 1e-9)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch}] Average Loss = {avg_loss:.4f}")
        
        metrics = evaluate(model, val_loader, device, ks=[1, 5, 10, 15, 20])
        metrics_str = ", ".join([f"@{k}: NDCG={m[0]:.4f}, HIT={m[1]:.4f}, MRR={m[2]:.4f}" for k, m in metrics.items()])
        print(f"[Val Epoch {epoch}] {metrics_str}")
        
        # Save checkpoint, excluding base_model
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
    saved_state = {k: v for k, v in state_dict.items() if not k.startswith("base_model")}
    ckpt_path = os.path.join(args.output_dir, f"trained_params_final.pt")
    torch.save(saved_state, ckpt_path)
    print(f"[Info] Final checkpoint saved to: {ckpt_path}")
    
    # Test set evaluation (already has tqdm in evaluate())
    metrics = evaluate(model, test_loader, device, ks=[1, 5, 10, 15, 20])
    for k, (ndcg, hit, mrr) in metrics.items():
        print(f"[Test] @ {k}: NDCG={ndcg:.4f}, HIT={hit:.4f}, MRR={mrr:.4f}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_losses)), train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    loss_curve_path = os.path.join(args.output_dir, "training_loss_curve.png")
    plt.savefig(loss_curve_path)
    print(f"[Info] Training loss curve saved to {loss_curve_path}")

def main():
    args = parse_args()
    main_worker(args.local_rank, args)

if __name__ == "__main__":
    main()

