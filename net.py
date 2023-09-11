from config import vec_size, dropout_p, block_size, d_k, d_model, n_embed, num_heads, num_blocks, num_classes, pre_training, N, P, C, batch_size
import torch
import torch.nn as nn
import torch.backends.cuda
import torch.nn.functional as F
from config import device, xavier, pos_sin, flash_att

torch.backends.cuda.enable_flash_sdp(enabled = True)

class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()
        position = torch.arange(block_size).unsqueeze(1)
        even_term = torch.pow(10000, torch.arange(0, vec_size, 2) / vec_size)
        even = torch.sin(position / even_term)
        odd = torch.cos(position / even_term)
        pe = torch.stack([even, odd], dim = 2)
        pe = pe.flatten(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x  = self.pe[:x.size(1)].unsqueeze(0)
        return x

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.queries = nn.Linear(vec_size, head_size, bias = False)
        self.keys = nn.Linear(vec_size, head_size, bias = False)
        self.values = nn.Linear(vec_size, head_size, bias = False)
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B,T,C = x.shape
        if flash_att:

            q = self.queries(x)
            k = self.keys(x)
            v = self.values(x)

            out = F.scaled_dot_product_attention(q,k , v)
        else:
            wei = torch.matmul(self.queries(x), self.keys(x).transpose(-2, -1)) * d_k ** -0.5
            wei = self.softmax(wei)
            wei = self.dropout(wei)

            out = torch.matmul(wei, self.values(x))

        return out
    
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.multihead = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, n_embed)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.multihead], dim = -1)
        return self.dropout(self.proj(out))
    
class MLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ffn = nn.Sequential(
                    nn.Linear(n_embed, 4 * n_embed),
                    nn.ReLU(),
                    nn.Linear(4 * n_embed, n_embed),
                    nn.Dropout(dropout_p)
                    )
        
    def forward(self, x):
        return self.ffn(x)
    
class Block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sa = MultiHead(num_heads, d_k)
        self.ffm = MLP()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x =  x + self.sa(self.ln1(x))
        x = x + self.ffm(self.ln2(x))
        return x

class PreTrainingMLP(nn.Module):
    def __init__(self):
        super(PreTrainingMLP, self).__init__()
        self.hidden_layer = nn.Linear(vec_size, vec_size)
        self.output_layer = nn.Linear(vec_size, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        logits = self.output_layer(x)
        return logits

class PatchEmbedding(nn.Module):
    def __init__(self):
        super(PatchEmbedding, self).__init__()
        self.num_patches = N+1
        self.patch_embed_dim = C * P**2 
        self.vec_dim = vec_size
        
        self.patch_embeddings = nn.ModuleList([
            nn.Linear(self.patch_embed_dim, self.vec_dim) for _ in range(self.num_patches)
        ])
    
    def forward(self, x):
        # x is a tensor of shape (batch_size, num_patches, patch_embed_dim)
        embeddings = torch.stack([self.patch_embeddings[i](x[:, i]) for i in range(self.num_patches)], dim=1)
        return embeddings


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.blocks = nn.ModuleList([Block() for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(n_embed)
        if pre_training:
            self.class_FC = PreTrainingMLP()
        self.patch_encoding = PatchEmbedding()
        if xavier:
            self.cls = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, P*P*C))).to(device)
        else:
            self.cls = nn.Parameter(torch.rand((1, 1, P*P*C))*0.1).to(device)
        if pos_sin:
            self.positional_encoder = PositionalEncoding()
        else:
            self.positional_encoder = nn.Embedding(block_size, vec_size) 
    def forward(self, x):
        x = torch.cat([self.cls.repeat(x.shape[0], 1 , 1), x], dim = 1)
        if pos_sin:
            x = self.patch_encoding(x) + self.positional_encoder(x).to(device)
        else:
            x = self.patch_encoding(x) + self.positional_encoder(torch.arange(block_size).to(device))

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        out = self.class_FC(x[:, 0, :])
        
        return out
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

