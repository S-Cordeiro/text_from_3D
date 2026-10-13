import torch
import torch.nn as nn
from .blocks import SelfAttention, CrossAttention, MLP

class ObjectDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.odconfig = config.object_decoder

        # self attention
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.self_attn = SelfAttention(config.d_model, self.odconfig.n_head)

        # cross attention
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.cross_attn = CrossAttention(config.d_model, self.odconfig.n_head)

        self.ln_3 = nn.LayerNorm(config.d_model)

        self.ffwd = MLP(config.d_model)

    def forward(self, queries, visual_tokens):

        x = self.ln_1(queries)
        attn = self.self_attn(x)
        queries = queries + attn

        x = self.ln_2(queries)
        attn = self.cross_attn(x, visual_tokens)

        queries = queries + attn

        queries = queries + self.ffwd(
            self.ln_3(queries)
        )

        return queries

class ObjectDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.odconfig = config.object_decoder
        self.n_objects = self.odconfig.n_objects

        self.object_queries = nn.Embedding(self.n_objects, config.d_model)
        self.blocks = nn.ModuleList([ObjectDecoderBlock(config) for _ in range(self.odconfig.n_layers)])

    def forward(self, visual_tokens):
        B = visual_tokens.shape[0]
        
        queries = self.object_queries.weight.unsqueeze(0)
        queries = queries.expand(B, -1, -1)

        for block in self.blocks:
            queries = block(queries, visual_tokens)
        
        return queries

class ObjectPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.odconfig = config.object_decoder

        self.class_head = nn.Linear(config.d_model, config.n_classes + 1)
        self.position_head = nn.Linear(config.d_model, 3)
        self.scale_head = nn.Linear(config.d_model, 3)
        self.rotation_head = nn.Linear(config.d_model, 6)
        
    def forward(self, object_queries):
        out = {
            "class_logits" : self.class_head(object_queries),
            "position" :     self.position_head(object_queries),
            "scale" :        self.scale_head(object_queries),
            "rotation" :     self.rotation_head(object_queries),
        }
        return out