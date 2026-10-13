import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.c_attn_q = nn.Linear(d_model, d_model)
        self.c_attn_k = nn.Linear(d_model, d_model)
        self.c_attn_v = nn.Linear(d_model, d_model)

        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, visual_tokens):

        B, Tq, C = queries.shape
        _, Tk, _ = visual_tokens.shape

        q = self.c_attn_q(queries)
        k = self.c_attn_k(visual_tokens)
        v = self.c_attn_v(visual_tokens)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v)

        # (B, n_head, Tq, head_dim) -> (B, Tq, C)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)

        return self.c_proj(y)


class SelfAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q = qkv[:, :, :self.d_model]
        k = qkv[:, :, self.d_model:2*self.d_model]
        v = qkv[:, :, 2*self.d_model:]

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)

        return out


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model*4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_model*4, d_model)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x