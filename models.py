import torch.nn as nn
import torch

from omegaconf import DictConfig
from einops import rearrange, repeat

# construct transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_head, num_layers, **args):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True), num_layers=num_layers)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = self.transformer(x_emb)
        x_emb = self.layernorm(x_emb)
        x_out = self.fc(x_emb[:, -1, :])  # only the last token of output is used
        return x_out

class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, n_heads: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model)
        )
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm (x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)

        return a2

class Transformer(torch.nn.Module):
    def __init__(self, num_tokens, num_layers, d_model, n_head, seq_len=5):
        super().__init__()

        self.token_embeddings = nn.Embedding(num_tokens, d_model)
        self.position_embeddings = nn.Embedding(seq_len, d_model)
        self.model = nn.Sequential(
            *[DecoderBlock(d_model, n_head) for _ in range(num_layers)],
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_tokens)
        )
    def forward(self, inputs):
        batch_size, context_len = inputs.shape

        token_embedding = self.token_embeddings(inputs)

        positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
        position_embedding = self.position_embeddings(positions)

        embedding = token_embedding + position_embedding

        embedding = rearrange(embedding, 'b s d -> s b d')

        return self.model(embedding)[-1, :, :]

# construct MLP model
class MLP(nn.Module):
    def __init__(self, num_tokens, output_dim, num_layers, embed_dim, hidden_dim, seq_len=5):
        super(MLP, self).__init__()

        self.token_embeddings = nn.Embedding(num_tokens, embed_dim)
        self.position_embeddings = nn.Embedding(seq_len, embed_dim)
        # fully connected layers
        layer_list = [nn.Linear(embed_dim * (seq_len - 1), hidden_dim),
                     nn.ReLU()]
        for _ in range(num_layers - 1):
            layer_list += [nn.Linear(hidden_dim, hidden_dim), 
                          nn.ReLU()]
        layer_list += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layer_list)

    def forward(self, inputs):
        batch_size, context_len = inputs.shape

        token_embedding = self.token_embeddings(inputs)

        positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
        position_embedding = self.position_embeddings(positions)

        embedding = token_embedding + position_embedding
        embedding = rearrange(embedding, 'b s d -> b (s d)')
        # forward
        return self.model(embedding)

# construct LSTM model    
class LSTMModel(nn.Module):
    def __init__(self, num_tokens, output_dim, num_layers, embed_dim, hidden_dim, seq_len=5):
        super(LSTMModel, self).__init__()

        self.token_embeddings = nn.Embedding(num_tokens, embed_dim)
        self.position_embeddings = nn.Embedding(seq_len, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        batch_size, context_len = inputs.shape

        token_embedding = self.token_embeddings(inputs)

        positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
        position_embedding = self.position_embeddings(positions)

        embedding = token_embedding + position_embedding
        lstm_out, _ = self.lstm(embedding)
        output = self.fc(lstm_out[:, -1, :])
        return output

def get_model(config: DictConfig):
    try: 
        K = config.K
    except:
        K = 2
    if config.model_type.lower() == 'transformer_native':
        model = TransformerModel(config.train.p+2, 
                                 config.train.p,
                                 **config.transformer)
    elif config.model_type.lower() == 'transformer':
        model = Transformer(config.train.p+2,
                             seq_len = 2 * K + 1,
                            **config.transformer)
    elif config.model_type.lower() == 'mlp':
        model = MLP(config.train.p+2,
                    config.train.p, 
                    seq_len = 2 * K + 1,
                    **config.mlp)
    elif config.model_type.lower() == 'lstm':
        model = LSTMModel(config.train.p+2,
                          config.train.p,
                          seq_len = 2 * K + 1,
                          **config.lstm)
    else:
        raise ValueError(f"The model_type {config.model_type} is not supported!")
    return model