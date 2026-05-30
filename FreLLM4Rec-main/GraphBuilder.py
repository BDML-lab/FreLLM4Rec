import torch
import os
from tqdm import tqdm
import numpy as np

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix

def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

class ItemKNNGraphBuilder:
    def __init__(self, model, item_ids, output_dir, device, topk=10, norm_type='sym'):
        self.model = model
        self.item_ids = item_ids
        self.num_items = len(item_ids)
        self.output_dir = output_dir
        self.device = device
        self.topk = topk
        self.norm_type = norm_type
        self.knn_graph = None
        self.knn_graph_path = os.path.join(output_dir, f"knn_graph_topk{topk}_{norm_type}.pt")

        self.knn_graph_id = None
        self.knn_graph_path_id = os.path.join(output_dir, f"knn_graph_topk{topk}_id_{norm_type}.pt")

    def build_knn_graph(self, force_rebuild=False):
        if not force_rebuild and os.path.exists(self.knn_graph_path):
            print(f"[Info] Loading cached KNN graph from {self.knn_graph_path}")
            self.knn_graph = torch.load(self.knn_graph_path, map_location=self.device)
            return self.knn_graph

        print(f"[Info] Building Item-Item KNN graph with topk={self.topk}, norm_type={self.norm_type}")
        embeddings = self.model.get_text_embeddings(self.item_ids, self.device)
        embeddings = embeddings.to(self.device)  # [num_items, token_dim]
        
        sim_matrix = build_sim(embeddings)  # [num_items, num_items]

        self.knn_graph = build_knn_normalized_graph(
            sim_matrix, self.topk, is_sparse=True, norm_type=self.norm_type
        )

        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.knn_graph, self.knn_graph_path)
        print(f"[Info] KNN graph saved to {self.knn_graph_path}")
        return self.knn_graph

    def build_knn_graph_id(self, force_rebuild=False, df_test=None, item_num=None):
        if not force_rebuild and os.path.exists(self.knn_graph_path_id):
            print(f"[Info] Loading cached KNN_ID graph from {self.knn_graph_path_id}")
            self.knn_graph_id = torch.load(self.knn_graph_path_id, map_location=self.device)
            return self.knn_graph_id
        
        # 构造 UI 矩阵用于图傅里叶变换
        print("[Info] Constructing UI matrix for Graph Fourier Transform...")
        user_interactions = [list(filter(lambda x: x != 0, seq)) for seq in df_test["seq"]]
        N_users = len(user_interactions)
        unique_items = set().union(*user_interactions)
        unique_items = list(range(1, item_num + 1))
        if 0 in unique_items:
            unique_items.discard(0)
        sorted_item_ids = sorted(list(unique_items))
        M_items = len(sorted_item_ids)
        print(f"[Info] UI matrix shape: Users={N_users}, Items={M_items}")

        UI = np.zeros((N_users, M_items), dtype=np.float32)
        item_id_to_col = {item_id: idx for idx, item_id in enumerate(sorted_item_ids)}
        for i, seq in enumerate(user_interactions):
            for item in seq:
                if item in item_id_to_col:
                    UI[i, item_id_to_col[item]] = 1.0
        
        print("[Info] Computing Item-Item co-occurrence matrix...")
        knn_graph_id = np.dot(UI.T, UI)  # shape: (M_items, M_items)
        
        adj = torch.from_numpy(knn_graph_id).float()
        
        print(f"[Info] Converting to sparse Laplacian with norm_type={self.norm_type}")
        indices = torch.nonzero(adj, as_tuple=False).t()
        values = adj[indices[0], indices[1]]
        edge_index, edge_weight = get_sparse_laplacian(indices, values, num_nodes=adj.shape[0], normalization=self.norm_type)
        self.knn_graph_id = torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape, device=self.device)
        
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.knn_graph_id, self.knn_graph_path_id)
        print(f"[Info] KNN_ID graph saved to {self.knn_graph_path_id}")
        return self.knn_graph_id

    def get_knn_graph(self):
        if self.knn_graph is None:
            self.build_knn_graph()
        return self.knn_graph
    
    def get_knn_graph_id(self):
        if self.knn_graph_id is None:
            self.build_knn_graph_id()
        return self.knn_graph_id
    

    def apply_laplacian_operation(self, operation='sym'):
        if self.knn_graph is None:
            self.build_knn_graph()
        
        # 将稀疏矩阵转换为稠密矩阵进行操作（若需要）
        adj = self.knn_graph.to_dense()
        laplacian = get_dense_laplacian(adj, normalization=operation)
        
        # 转换回稀疏格式
        indices = torch.nonzero(laplacian, as_tuple=False).t()
        values = laplacian[indices[0], indices[1]]
        return torch.sparse_coo_tensor(indices, values, laplacian.shape)

    def get_nearest_neighbors(self, item_id, k=None):
        if self.knn_graph is None:
            self.build_knn_graph()
        
        item_idx = self.item_ids.index(item_id)
        row = self.knn_graph.to_dense()[item_idx]
        knn_val, knn_ind = torch.topk(row, k or self.topk, dim=-1)
        neighbor_indices = knn_ind[knn_val > 0]  # 过滤掉零值
        return torch.tensor([self.item_ids[i] for i in neighbor_indices], device=self.device)
 