# models/projector.py
import torch.nn as nn
import torch
import torch.nn.init as init

class MlpProjector(nn.Module):
    def __init__(self, token_dim, proj_intermediate_dim, out_dim):
        """
        两层 MLP 投影: token_dim -> proj_intermediate_dim -> out_dim
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, proj_intermediate_dim),
            nn.ReLU(),
            nn.Linear(proj_intermediate_dim, out_dim)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Xavier 均匀分布初始化
                init.xavier_uniform_(m.weight)
                # 如果有偏置，将其初始化为 0
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
    
class FilterLayer(nn.Module):
    def __init__(self, max_seq_len, hidden_size, hidden_dropout_prob, cutoff_freq_ratio=0.1):
        super(FilterLayer, self).__init__()
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.hidden_size = hidden_size  # 隐藏层维度
        self.freq_dim = max_seq_len // 2 + 1  # FFT后的频率维度
        self.cutoff_freq_ratio = min(cutoff_freq_ratio, 1.0)  # 截止频率比例
        self.cutoff_freq_ratio = max(cutoff_freq_ratio, 0.1)  # 截止频率比例


        
        # 初始化Butterworth低通滤波掩码
        self.register_buffer("low_pass_mask", self._create_low_pass_mask())
        
        self.out_dropout = nn.Dropout(hidden_dropout_prob)  # Dropout层

    def _create_low_pass_mask(self):
        """创建Butterworth低通滤波掩码"""
        freq_indices = torch.arange(self.freq_dim, dtype=torch.float32)  # 频率索引
        f_k = freq_indices / (self.freq_dim - 1)  # 归一化频率
        cutoff_freq = int(self.freq_dim * self.cutoff_freq_ratio)  # 截止频率索引
        f_c = cutoff_freq / (self.freq_dim - 1)  # 归一化截止频率
        n = 2  # 二阶Butterworth滤波器
        with torch.no_grad():
            if f_c == 0:  # 处理截止频率为0的边缘情况
                H = torch.zeros(self.freq_dim)
                H[0] = 1  # 仅保留DC分量
            else:
                H = 1 / torch.sqrt(1 + (f_k / f_c).pow(2 * n))  # Butterworth公式
        return H.view(1, self.freq_dim, 1).expand(1, self.freq_dim, self.hidden_size)

    def _create_dynamic_low_pass_mask(self, freq_dim):
        """动态创建Butterworth低通滤波掩码，适应不同序列长度"""
        freq_indices = torch.arange(freq_dim, dtype=torch.float32)
        f_k = freq_indices / (freq_dim - 1)
        cutoff_freq = int(freq_dim * self.cutoff_freq_ratio)
        f_c = cutoff_freq / (freq_dim - 1)
        n = 2
        with torch.no_grad():
            if f_c == 0:
                H = torch.zeros(freq_dim)
                H[0] = 1
            else:
                H = 1 / torch.sqrt(1 + (f_k / f_c).pow(2 * n))
        return H.view(1, freq_dim, 1).expand(1, freq_dim, self.hidden_size)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.size()  # 输入张量形状
        freq_dim = seq_len // 2 + 1  # 当前序列的频率维度
        
        # 执行实数到复数的FFT
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')  # 形状: [batch, freq_dim, hidden]
        
        # 动态调整掩码以适应序列长度
        if freq_dim != self.freq_dim:
            current_mask = self._create_dynamic_low_pass_mask(freq_dim).to(x.device)
        else:
            current_mask = self.low_pass_mask[:, :freq_dim, :].to(x.device)
        
        # 应用低通滤波
        x = x * current_mask
        
        # 逆FFT回到时域
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        
        # 应用Dropout和层归一化
        hidden_states = self.out_dropout(sequence_emb_fft)
        return hidden_states



# import torch
# import torch.nn as nn

# class FilterLayer(nn.Module):
#     def __init__(self, max_seq_len, hidden_size, hidden_dropout_prob, cutoff_freq_ratio=0.9):
#         super(FilterLayer, self).__init__()
#         self.max_seq_len = max_seq_len
#         self.hidden_size = hidden_size
#         self.freq_dim = max_seq_len // 2 + 1
#         self.cutoff_freq_ratio = min(cutoff_freq_ratio, 1.0)
#         self.cutoff_freq_ratio = max(cutoff_freq_ratio, 0.0)

#         # 初始化Butterworth高通滤波掩码
#         self.register_buffer("high_pass_mask", self._create_high_pass_mask())
        
#         self.out_dropout = nn.Dropout(hidden_dropout_prob)

#     def _create_high_pass_mask(self):
#         """创建Butterworth高通滤波掩码"""
#         freq_indices = torch.arange(self.freq_dim, dtype=torch.float32)
#         f_k = freq_indices / (self.freq_dim - 1)  # 归一化频率
#         cutoff_freq = int(self.freq_dim * self.cutoff_freq_ratio)  # 截止频率索引
#         f_c = cutoff_freq / (self.freq_dim - 1)  # 归一化截止频率
#         n = 2  # 二阶Butterworth滤波器
#         with torch.no_grad():
#             if f_c == 0:  # 处理截止频率为0的边缘情况
#                 H = torch.ones(self.freq_dim)  # 保留所有频率
#             else:
#                 # 高通滤波公式：H(f) = 1 / sqrt(1 + (f_c/f)^(2n))
#                 H = torch.ones(self.freq_dim)
#                 mask = f_k > 0  # 避免 f_k = 0 时除零
#                 H[mask] = 1 / torch.sqrt(1 + (f_c / f_k[mask]).pow(2 * n))
#                 H[0] = 0  # 显式设置 DC 分量为 0（高通滤波去除直流分量）
#         return H.view(1, self.freq_dim, 1).expand(1, self.freq_dim, self.hidden_size)

#     def _create_dynamic_high_pass_mask(self, freq_dim):
#         """动态创建Butterworth高通滤波掩码，适应不同序列长度"""
#         freq_indices = torch.arange(freq_dim, dtype=torch.float32)
#         f_k = freq_indices / (freq_dim - 1)
#         cutoff_freq = int(freq_dim * self.cutoff_freq_ratio)
#         f_c = cutoff_freq / (freq_dim - 1)
#         n = 2
#         with torch.no_grad():
#             if f_c == 0:
#                 H = torch.ones(freq_dim)
#             else:
#                 H = torch.ones(freq_dim)
#                 mask = f_k > 0
#                 H[mask] = 1 / torch.sqrt(1 + (f_c / f_k[mask]).pow(2 * n))
#                 H[0] = 0
#         return H.view(1, freq_dim, 1).expand(1, freq_dim, self.hidden_size)

#     def forward(self, input_tensor):
#         batch, seq_len, hidden = input_tensor.size()
#         freq_dim = seq_len // 2 + 1
        
#         # 执行实数到复数的FFT
#         x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        
#         # 动态调整掩码以适应序列长度
#         if freq_dim != self.freq_dim:
#             current_mask = self._create_dynamic_high_pass_mask(freq_dim).to(x.device)
#         else:
#             current_mask = self.high_pass_mask[:, :freq_dim, :].to(x.device)
        
#         # 应用高通滤波
#         x = x * current_mask
        
#         # 逆FFT回到时域
#         sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        
#         # 应用Dropout
#         hidden_states = self.out_dropout(sequence_emb_fft)
#         return hidden_states
    

class Qwen2LayerWithFilter(nn.Module):
    def __init__(self, original_layer, filter_layer):
        super(Qwen2LayerWithFilter, self).__init__()
        self.original_layer = original_layer  # Original Qwen2 Transformer layer
        self.filter_layer = filter_layer      # Added FilterLayer

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        # Pass through original Qwen2 layer
        layer_outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        # layer_outputs[0] is the hidden states, apply FilterLayer
        filtered_hidden_states = self.filter_layer(layer_outputs[0])
        # Replace the original hidden states with filtered ones
        return (filtered_hidden_states,) + layer_outputs[1:]  # Preserve other outputs (e.g., attentions, past_key_value)
    