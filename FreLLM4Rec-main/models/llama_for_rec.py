import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from models.projector import MlpProjector, FilterLayer, Qwen2LayerWithFilter
import torch.sparse
import torch.linalg as linalg
class LlamaForRec(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.item_num = hparams.item_num
        self.embed_mode = hparams.embed_mode
        self.max_seq_len = hparams.max_seq_len


        self.num_check = getattr(hparams, 'num_check', 9)
        dropout_prob = getattr(hparams, 'dropout_prob', 0.2)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fusion_dropout = nn.Dropout(p=dropout_prob)

        from Pre_Train_Rec_Model.sasrec.model import SASRec
        kwargs, checkpoint = torch.load(hparams.rec_model_path, map_location="cpu")
        self.rec_model = SASRec(**kwargs)
        self.rec_model.load_state_dict(checkpoint)

        if getattr(hparams, "new_item_emb_path", None):
            new_emb = torch.load(hparams.new_item_emb_path, map_location="cpu")  
            if isinstance(new_emb, dict):        
                new_emb = new_emb["weight"]

            new_num, new_dim = new_emb.shape
            rec_num, rec_dim = self.rec_model.item_emb.weight.shape

            if (new_num, new_dim) != (rec_num, rec_dim):
                self.rec_model.item_emb = nn.Embedding.from_pretrained(new_emb, freeze=False)

                self.item_num = new_num - 1
                self.hparams.item_num = new_num - 1

                print(f"[Info] item_emb 层已重建，shape = {new_emb.shape}")
            else:
                with torch.no_grad():
                    self.rec_model.item_emb.weight.copy_(new_emb)
                print(f"[Info] item_emb 权重已覆盖，shape = {new_emb.shape}")

        self.rec_model.eval()
        for p in self.rec_model.parameters():
            p.requires_grad = False

        self.base_model = AutoModelForCausalLM.from_pretrained(hparams.pretrained_model)
        self.token_dim = self.base_model.config.hidden_size



        if hasattr(hparams, 'text_embeddings') and hparams.text_embeddings is not None:
            if hparams.text_embeddings.shape[0] != self.item_num:
                raise ValueError(f"text_embeddings大小 ({hparams.text_embeddings.shape[0]}) 不匹配 item_num ({self.item_num})")
            if hparams.text_embeddings.shape[1] != self.token_dim:
                raise ValueError(f"text_embeddings维度 ({hparams.text_embeddings.shape[1]}) 不匹配 token_dim ({self.token_dim})")
            self.register_buffer("text_embeddings", hparams.text_embeddings)
        else:
            self.text_embeddings = None
            print("[警告] 未提供预处理的text_embeddings，将在需要时动态计算")

        self.out_type = getattr(hparams, 'out_type', 0)
        if self.out_type == 2:
            self.out_proj = MlpProjector(token_dim=self.token_dim, 
                                         proj_intermediate_dim=hparams.proj_intermediate_dim, 
                                         out_dim=self.token_dim)
        elif self.out_type == 1:
            self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        else:
            self.out_proj = nn.Identity()

        self.mlp_type = getattr(hparams, 'mlp_type', 2)
        if self.mlp_type == 2:
            self.item_proj = MlpProjector(token_dim=self.token_dim, 
                                         proj_intermediate_dim=hparams.proj_intermediate_dim, 
                                         out_dim=self.token_dim)
        elif self.mlp_type == 1:
            self.item_proj = nn.Linear(self.token_dim, self.token_dim)
        else:
            self.item_proj = nn.Identity()
        
        



        self.id_proj = nn.Linear(self.rec_model.item_emb.embedding_dim, self.token_dim)

        if hparams.num_trainable_items > 0:
            self.train_item_emb = nn.Embedding(hparams.num_trainable_items, self.token_dim)
            init.normal_(self.train_item_emb.weight, mean=0.0, std=0.1)
        else:
            self.train_item_emb = None

        self.text_proj = nn.Linear(self.token_dim, self.rec_model.item_emb.embedding_dim)

        self.fusion_lin = nn.Linear(2 * self.token_dim, self.token_dim)
        self.graph_conv_weight = getattr(hparams, 'graph_conv_weight', 0.3)
        self.item_fuse_proj = nn.Linear(self.token_dim, self.token_dim)
        self.gate_t = nn.Sequential(
            MlpProjector(token_dim=self.token_dim, 
                        proj_intermediate_dim=hparams.proj_intermediate_dim, 
                        out_dim=self.token_dim),
            nn.Sigmoid()
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.ln_f = nn.LayerNorm(self.token_dim)

        self.knn_graph = None
        self.knn_graph_id = None

        self.id_layernorm = nn.LayerNorm(self.token_dim)
        self.text_layernorm = nn.LayerNorm(self.token_dim)

        print(self.item_num)
        print(self.rec_model.item_emb.weight.shape)
        all_item_ids_tensor = torch.arange(1, self.item_num + 1, dtype=torch.long)
        self.register_buffer('base_id_embeddings', self.rec_model.item_emb(all_item_ids_tensor))
        
        self.num_hidden_layers = self.base_model.config.num_hidden_layers
        if hparams.inject_every_layer:
            print("注入all layer")
            for i in range(self.num_hidden_layers):
                original_layer = self.base_model.model.layers[i]
                filter_layer = FilterLayer(max_seq_len = hparams.max_seq_len, hidden_size = self.token_dim, hidden_dropout_prob = dropout_prob, cutoff_freq_ratio = hparams.init_fa)
                self.base_model.model.layers[i] = Qwen2LayerWithFilter(original_layer, filter_layer)

        for name, param in self.base_model.named_parameters():
            if "lora" not in name and "filter_layer" not in name:
                param.requires_grad = False

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def get_rec_embedding(self, item_ids, device):
        tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
        rec_emb = self.rec_model.item_emb(tensor)
        return rec_emb

    def get_text_embeddings(self, item_ids, device):
        if self.text_embeddings is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, dtype=torch.long, device=device)
            text_emb = self.text_embeddings[item_ids - 1]
            return text_emb
        else:
            print("[警告] 无预处理的text_embeddings，使用动态计算")
            embeddings = []
            for item_id in item_ids:
                title = self.hparams.id2name.get(item_id, str(item_id)) if hasattr(self.hparams, 'id2name') else str(item_id)
                inputs = self.hparams.tokenizer(title, return_tensors="pt", truncation=True).to(device)
                input_ids = inputs["input_ids"]
                token_embeds = self.base_model.get_input_embeddings()(input_ids)
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                masked_token_embeds = token_embeds * attention_mask.unsqueeze(-1)
                sum_embeds = masked_token_embeds.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)
                mean_pooling_embeds = sum_embeds / sum_mask
                embeddings.append(mean_pooling_embeds.squeeze(0))
            text_emb = torch.stack(embeddings, dim=0)
            return text_emb

    def get_trainable_embedding(self, item_ids, device):
        if self.train_item_emb is None:
            raise ValueError("未启用可训练嵌入，请设置num_trainable_items > 0")
        return self.train_item_emb(torch.tensor(item_ids, device=device))

    def get_text_embedding(self, item_ids, device):
        text_emb = self.get_text_embeddings(item_ids, device)
        return self.text_proj(text_emb)

    def get_all_item_embeddings(self, device):
        all_item_ids = list(range(1, self.item_num + 1))
        if self.knn_graph_id is None:
            raise ValueError("KNN图未初始化")

        id_emb = self.id_proj(self.base_id_embeddings)
        
        fused_neighbor = torch.sparse.mm(self.knn_graph_id, id_emb)
        id_emb = (1 - self.graph_conv_weight) * id_emb + self.graph_conv_weight * fused_neighbor
        id_emb = self.id_layernorm(id_emb)
        
        text_emb = self.item_proj(self.text_embeddings) 
        fused_neighbor = torch.sparse.mm(self.knn_graph_id, text_emb)
        text_emb = (1 - self.graph_conv_weight) * text_emb + self.graph_conv_weight * fused_neighbor
        text_emb = self.text_layernorm(text_emb)
        
        combined = torch.cat([id_emb, text_emb], dim=-1)
        gate = torch.sigmoid(self.fusion_lin(combined))
        fused = gate * id_emb + (1 - gate) * text_emb
        emb = fused
        return emb

    def get_hidden_state(self, model, seq_emb, attention_mask):
        outputs = model.base_model(
            inputs_embeds=seq_emb,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        res = [self.out_proj(x) for x in outputs.hidden_states]
        return res

    def forward(self, seqs, next_items, negatives):
        device = next(self.parameters()).device
        all_item_emb = self.get_all_item_embeddings(device)  # [item_num, token_dim]

        if self.training:
            seqs_t = [torch.tensor(s, device=device, dtype=torch.long) for s in seqs]
            next_items_t = torch.tensor(next_items, device=device, dtype=torch.long)
            
            train_seqs = [torch.cat((s[s!=0], n.unsqueeze(0))) for s, n in zip(seqs_t, next_items_t)]
            
            seq_lengths = torch.tensor([len(s) for s in train_seqs], device=device, dtype=torch.long)
            seq_embeddings = [all_item_emb[s - 1] for s in train_seqs]
            
            batch_padded_variable_len = nn.utils.rnn.pad_sequence(seq_embeddings, batch_first=True, padding_value=0.0)
            
            current_len = batch_padded_variable_len.size(1)
            target_len = self.max_seq_len
            if current_len > target_len:
                batch_padded = batch_padded_variable_len[:, :target_len]
            else:
                pad_width = target_len - current_len
                batch_padded = nn.functional.pad(batch_padded_variable_len, (0, 0, 0, pad_width), 'constant', 0)

            effective_lengths = torch.min(seq_lengths, torch.tensor(self.max_seq_len, device=device))
            attn_masks = torch.arange(self.max_seq_len, device=device)[None, :] < effective_lengths[:, None]

            hidden_states = self.get_hidden_state(self.base_model, batch_padded, attn_masks)[self.hparams.hidden_layer]
            hidden_states = self.ln_f(hidden_states) # [B, max_seq_len, D]
            pred_vectors = hidden_states[:, :-1, :] # [B, max_seq_len-1, D]

            pos_embs = batch_padded[:, 1:, :] # [B, max_seq_len-1, D]

            negatives_t = torch.tensor(negatives, device=device, dtype=torch.long) # [B, Num_Neg]
            neg_embs = all_item_emb[negatives_t - 1] # [B, Num_Neg, D]

            pos_logits = (pred_vectors * pos_embs).sum(dim=-1) # [B, max_seq_len-1]

            neg_logits = torch.bmm(pred_vectors, neg_embs.transpose(1, 2))

            logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)

            labels = torch.zeros_like(pos_logits, dtype=torch.long)
            
            loss_mask = attn_masks[:, 1:] # [B, max_seq_len-1]
            
            if loss_mask.sum() == 0: # 避免在所有都是padding的情况下除以0
                return torch.tensor(0.0, device=device)

            loss = self.loss_fn(logits[loss_mask], labels[loss_mask])
            
            return loss
            # =================================================================
        else:
            seqs_t = [torch.tensor(s, device=device, dtype=torch.long) for s in seqs]
            valid_seqs = [s[s!=0] for s in seqs_t]
            seq_lengths = torch.tensor([len(s) if len(s) > 0 else 1 for s in valid_seqs], device=device, dtype=torch.long)
            
            seq_embeddings = [all_item_emb[s - 1] if len(s)>0 else torch.zeros(1, self.token_dim, device=device) for s in valid_seqs]
            batch_padded = nn.utils.rnn.pad_sequence(seq_embeddings, batch_first=True, padding_value=0.0)[:, :self.max_seq_len]

            effective_lengths = torch.min(seq_lengths, torch.tensor(self.max_seq_len, device=device))
            attn_masks = torch.arange(self.max_seq_len, device=device)[None, :] < effective_lengths[:, None]

            hidden_states = self.get_hidden_state(self.base_model, batch_padded, attn_masks)[self.hparams.hidden_layer]
            hidden_states = self.ln_f(hidden_states) # [B, L, D]

            last_token_indices = (effective_lengths - 1).view(-1, 1, 1).expand(-1, -1, self.token_dim)
            pred_vectors = hidden_states.gather(1, last_token_indices).squeeze(1) # [B, D]

            logits_list = []
            for i in range(len(seqs)):
                pos_target_id = next_items[i]
                neg_target_ids = negatives[i]
                
                candidate_ids = torch.tensor([pos_target_id] + neg_target_ids, device=device, dtype=torch.long)
                candidate_embs = all_item_emb[candidate_ids - 1] # [1+Num_Neg, D]
                
                logits = candidate_embs @ pred_vectors[i] # [1+Num_Neg]
                logits_list.append(logits)

            return logits_list
