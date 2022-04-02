import copy
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from .modelBase import ModelBase


class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

        self.out = nn.Linear(768, 768)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(768, 3072) #hidden_size, mlp_dim
        self.fc2 = nn.Linear(3072, 768) #mlp_dim, hidden_size
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(0.01)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(ModelBase):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, parameters: dict):
        super(Embeddings, self).__init__(parameters)
        img_size = _pair(img_size)


        small_patch_size = _pair((self.base_filters, self.base_filters)) #16,16
        n_patches = (img_size[0] // small_patch_size[0]) * (img_size[1] // small_patch_size[1]) #1024

        self.patch_embeddings = self.Conv(in_channels=self.n_channels,
                                       out_channels=768,
                                       kernel_size=small_patch_size,
                                       stride=small_patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, vis):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

   
class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(768, eps=1e-6)
        for _ in range(12):  #12 layers
            layer = Block(vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(ModelBase):
    def __init__(self, img_size, vis, parameters: dict):
        super(Transformer, self).__init__(parameters)
        self.embeddings = Embeddings(img_size=img_size, parameters=parameters)
        self.encoder = Encoder(vis)


    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class DecoderBlock(ModelBase):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels,
            parameters: dict,):
        super().__init__(parameters)
                
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        
        self.double_conv = nn.Sequential(
            self.Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            self.Norm(out_channels),
            nn.ReLU(inplace=True),
            self.Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            self.Norm(out_channels),
            nn.ReLU(inplace=True)
        )

        
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.double_conv(x)
        return x

class DecoderCup(ModelBase):
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        head_channels = self.base_filters * 32 #512
        
        self.conv_more = nn.Sequential(
            self.Conv(768, head_channels, kernel_size=3, padding=1, bias=False),
            self.Norm(head_channels),
            nn.ReLU(inplace=True),
        )

        decoder_channels = (self.base_filters * 16, self.base_filters * 8, self.base_filters * 4, self.base_filters) #256,128,64,16
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.n_skip = 0
        skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, parameters=parameters) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class TransUNet(ModelBase):
    def __init__(self,  parameters: dict):
        super(TransUNet, self).__init__(parameters)
        self.zero_head = False
        self.classifier = 'seg'
        self.transformer = Transformer(img_size=self.patch_size, vis = False, parameters=parameters)
        self.decoder = DecoderCup(parameters=parameters)
        
        self.segmentation_head = nn.Sequential(
            self.Conv(self.base_filters, self.n_classes, kernel_size=3, padding=1),
            nn.Identity()     #self.Upsampling(scale_factor=upsampling) if upsampling > 1        
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits