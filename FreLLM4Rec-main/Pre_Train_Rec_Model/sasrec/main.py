import os
import time
import torch
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from model import SASRec
from data_preprocess import *
from utils import *
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# nohup python main.py --device=cuda --dataset Movies_and_TV



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=4, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

def precompute_text_embeddings(item_ids, tokenizer, base_model, id2name=None, save_path="text_embeddings.pt"):
    embeddings = {}
    base_model.to(args.device)
    
    with torch.no_grad():
        for item_id in tqdm(item_ids):
            title = id2name.get(item_id, str(item_id)) if id2name else str(item_id)
            inputs = tokenizer(title, return_tensors="pt", truncation=True).to(args.device)
            input_ids = inputs["input_ids"]
            token_embeds = base_model.get_input_embeddings()(input_ids)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            masked_token_embeds = token_embeds * attention_mask.unsqueeze(-1)
            sum_embeds = masked_token_embeds.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)
            mean_pooling_embeds = sum_embeds / sum_mask
            embeddings[item_id] = mean_pooling_embeds.squeeze(0).cpu()
    
    torch.save(embeddings, save_path)
    print(f"Text embeddings saved to {save_path}")

def analyze_embeddings_user_item_subgraph(model, dataset, args):
    """
    Analyzes the Graph Fourier spectrum of user embeddings on item subgraphs for each user.
    Constructs a k x k adjacency matrix from the full I-I graph, computes the spectrum,
    and averages the energy in 10% frequency bands across users for each layer.
    """
    output_dir = os.path.join('checkpoint', args.dataset, 'figure_user_item_subgraph')
    os.makedirs(output_dir, exist_ok=True)

    print("[Info] Starting user-item subgraph spectrum analysis...")

    # Step 1: Construct full I-I co-occurrence matrix
    print("[Info] Constructing full I-I co-occurrence matrix...")
    [train, valid, test, usernum, itemnum] = dataset
    user_interactions = [train[u] + valid[u] + test[u] for u in range(1, usernum + 1) if len(train[u]) > 0]
    unique_items = set().union(*user_interactions)
    unique_items = list(range(1, itemnum + 1))
    if 0 in unique_items:
        unique_items.discard(0)
    sorted_item_ids = sorted(list(unique_items))
    M_items = len(sorted_item_ids)
    
    II = np.zeros((M_items, M_items), dtype=np.float32)
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(sorted_item_ids)}
    
    for seq in user_interactions:
        valid_items = [item for item in seq if item in item_id_to_idx]
        for i in range(len(valid_items)):
            for j in range(i + 1, len(valid_items)):
                idx_i = item_id_to_idx[valid_items[i]]
                idx_j = item_id_to_idx[valid_items[j]]
                II[idx_i, idx_j] += 1
                II[idx_j, idx_i] += 1

    print(f"[Info] Full I-I matrix shape: Items={M_items}x{M_items}")

    # Step 2: Initialize storage for frequency band energies per layer
    num_layers = args.num_blocks+1  # Include input embedding + each block
    num_bands = 10  # 0-10%, 10-20%, ..., 90-100%
    band_energies_by_layer = {layer_idx: [[] for _ in range(num_bands)] for layer_idx in range(num_layers)}

    # Step 3: Process each user's sequence
    print("[Info] Extracting user embeddings and computing subgraph spectra...")
    model.eval()
    for u in tqdm(range(1, usernum + 1), desc="Processing user sequences"):
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # Prepare sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        valid_seq = [it for it in seq if it != 0]
        if len(valid_seq) < 2:  # Need at least 2 items for a graph
            continue

        # Create k x k subgraph adjacency matrix
        k = len(valid_seq)
        valid_items = [item for item in valid_seq if item in item_id_to_idx]
        if len(valid_items) < 2:
            continue
        valid_items.pop(0)
        valid_items.append(test[u][0])
        subgraph_adj = np.zeros((k, k), dtype=np.float32)
        for i in range(len(valid_items)):
            for j in range(i + 1, len(valid_items)):
                idx_i = item_id_to_idx[valid_items[i]]
                idx_j = item_id_to_idx[valid_items[j]]
                subgraph_adj[i, j] = II[idx_i, idx_j]
                subgraph_adj[j, i] = II[idx_j, idx_i]

        # Compute normalized Laplacian for subgraph
        D = np.diag(np.sum(subgraph_adj, axis=1))
        L = D - subgraph_adj
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(subgraph_adj, axis=1) + 1e-9))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        eigenvals, eigenvecs = np.linalg.eigh(L_norm)
        if len(eigenvals) == 0:
            print(f"[Warning] Empty eigendecomposition for sequence length {k}, skipping...")
            continue

        # Get hidden states for this sequence
        seq_tensor = np.expand_dims(seq, axis=0)
        seq_tensor = np.array(seq_tensor)
        hidden_states = model.log2feats(seq_tensor, return_all_layers=True)  # Modified to return all layers
        # print(type(hidden_states))
        # Compute Graph Fourier spectrum for each layer
        for layer_idx, hs in enumerate(hidden_states):
            user_vector = hs[0, ].cpu().detach().numpy()  # Last token as user representation

            # Graph Fourier Transform
            # Adjust user_vector dimension to match eigenvecs.T (k)
            k = eigenvecs.shape[0]  # Number of eigenvectors
            user_vector_dim = user_vector.shape[0]
            # print(user_vector)
            if user_vector_dim >= k:
                # Take first k dimensions
                user_vector_adjusted = user_vector[user_vector_dim - k:]
            else:
                # Pad with zeros if user_vector is smaller than k
                user_vector_adjusted = np.pad(user_vector, (0, k - user_vector_dim), mode='constant')

            f_hat = np.dot(eigenvecs.T, user_vector_adjusted)

            energy_spectrum = np.abs(f_hat)**2
            total_energy = np.sum(energy_spectrum) + 1e-9
            energy_spectrum_normalized = energy_spectrum / total_energy

            # Aggregate into 10% frequency bands
            num_eigenvals = len(eigenvals)
            band_size = max(1, num_eigenvals // num_bands)
            for band_idx in range(num_bands):
                start_idx = band_idx * band_size
                end_idx = (band_idx + 1) * band_size if band_idx < num_bands - 1 else num_eigenvals
                if start_idx >= num_eigenvals:
                    band_energy = 0.0
                else:
                    band_energy = np.sum(energy_spectrum_normalized[start_idx:end_idx])
                # print()
                # print(layer_idx)
                # print(band_idx)
                # print(band_energy)
                # print(np.array(band_energies_by_layer).shape)
                band_energies_by_layer[layer_idx][band_idx].append(band_energy)

    # Step 4: Compute average energies for each band and layer
    avg_band_energies = []
    for layer_idx in range(num_layers):
        layer_energies = []
        for band_idx in range(num_bands):
            energies = band_energies_by_layer[layer_idx][band_idx]
            if energies:
                avg_energy = np.mean(energies)
            else:
                avg_energy = 0.0
                print(f"[Warning] Layer {layer_idx}, Band {band_idx} has no valid energies")
            layer_energies.append(avg_energy)
        avg_band_energies.append(layer_energies)
        print(f"[Info] Layer {layer_idx} - Band Energies: {[f'{e:.4f}' for e in layer_energies]}")

    # Step 5: Visualize average band energies across layers
    plt.figure(figsize=(12, 6))
    band_labels = [f"{i*10}-{(i+1)*10}%" for i in range(num_bands)]
    for layer_idx in range(num_layers):
        plt.plot(band_labels, avg_band_energies[layer_idx], marker='o', label=f"Layer {layer_idx}")
    plt.title("Average Frequency Band Energies Across Layers (User-Specific Item Subgraph)")
    plt.xlabel("Frequency Band")
    plt.ylabel("Normalized Energy")
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join(output_dir, "user_item_subgraph_band_energies.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"[Info] Saved band energies plot to {plot_file}")

    # Step 6: Visualize average energy spectrum per layer
    print("[Info] Plotting average energy spectrum per layer...")
    for layer_idx in range(num_layers):
        avg_spectrum = np.zeros(num_bands)
        count = 0
        for band_idx in range(num_bands):
            energies = band_energies_by_layer[layer_idx][band_idx]
            if energies:
                avg_spectrum[band_idx] = np.mean(energies)
                count += 1
        if count == 0:
            print(f"[Warning] Layer {layer_idx} has no valid spectrum, skipping...")
            continue

        plt.figure(figsize=(8, 6))
        plt.plot(band_labels, avg_spectrum, marker='o', markersize=5)
        plt.title(f"Average Graph Fourier Energy Spectrum (User-Specific Item Subgraph) - Layer {layer_idx}")
        plt.xlabel("Frequency Band")
        plt.ylabel("Normalized Energy")
        plt.grid(True)
        plt.xticks(rotation=45)
        plot_file = os.path.join(output_dir, f"user_item_subgraph_energy_spectrum_layer_{layer_idx}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"[Info] Saved energy spectrum plot for layer {layer_idx} to {plot_file}")

if __name__ == '__main__':
    args.id2name = id2name

    args.token_dim = 3584

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    item_ids = list(range(0, itemnum + 1))

    print(len(user_test))
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    model.train()
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        # t_test = evaluate(model, dataset, args)
        # print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        analyze_embeddings_user_item_subgraph(model, dataset, args)
    
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only: break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
    
        if epoch % 40 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Quick validation (subset)', end='')
            t_valid = evaluate_valid(model, dataset, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f)' % (epoch, T, t_valid[0], t_valid[1]))
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = os.path.join("./checkpoint", args.dataset)
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(folder):
                try:
                    os.makedirs(folder)
                except Exception as e:
                    print(f"Error creating directory: {e}")
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
            # final test evaluation
            model.eval()
            print('Final test evaluation on full test set:', end='')
            t_test = evaluate(model, dataset, args)
            print(' NDCG@10: %.4f, HR@10: %.4f' % (t_test[0], t_test[1]))
    
    sampler.close()
    print("Done")