import torch 
import torch.nn as nn
import math
class FeedForwardBlock(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(FeedForwardBlock, self).__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=dropout)
  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size= vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

import torch
import torch.nn as nn
import math

## this is experimental change this if code doesn't work
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding matrix (1, max_sequence_length, d_model)
        encoding = torch.zeros(max_sequence_length, d_model) # gpt says start with because first dimension will have zeros
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        encoding = encoding.unsqueeze(0)  # Shape: (1, max_sequence_length, d_model)
        self.register_buffer('encoding', encoding)  # Makes it non-trainable

    def forward(self, x):
        # Add positional encoding to input and apply dropout
        x = x + self.encoding[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6):
        super().__init__()
        self.features=features
        self.eps=eps
        self.alpha = nn.Parameter(torch.ones(features)) # 512
        self.bias  =  nn.Parameter(torch.zeros(features)) # 512

    def forward(self, x):
        # inputs : 30 x 200 x 512
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

## Residual connection 
class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # sublayer is a function that is a nn module can be multihead block or feedforward block

## get the scaled dot product attention values 
class ScaledDotProduct(nn.Module): 
  def __init__(self):
    super(ScaledDotProduct, self).__init__()
    self.softmax = nn.Softmax(dim=-1)
  def forward(self, q, k, v, mask, dropout: nn.Dropout):
    d_k = q.shape[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(d_k)) # 30,8, 200, 200
    if mask is not None: 
        scaled.masked_fill_(mask == 0, -1e9)

    attention = self.softmax(scaled) # 30, 200, 64
    if dropout is not None:
        attention = dropout(attention)
    values = torch.matmul(attention, v)
    return values, attention

# simplified multi head attention
# class MultiHeadAttentionBlock(nn.Module):
#   def __init__(self, d_model, num_heads, dropout=False):
#     super().__init__()
#     self.d_model = d_model
#     self.num_heads = num_heads
#     assert d_model % num_heads == 0, " d_model is not divisible by num_heads"   
#     self.attention = ScaledDotProduct()
#     self.w_q = nn.Linear(d_model , d_model)
#     self.w_k = nn.Linear(d_model, d_model)
#     self.w_v = nn.Linear(d_model, d_model)
#     # self.qkv_layer = nn.Linear(d_model, 3 * d_model)
#     self.w_concat = nn.Linear(d_model, d_model)
#   def forward(self, q, k, v, mask=None):
#     # dot product with weight matrices
#     q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
#     # split tensor by number of heads
#     q, k, v = self.split(q), self.split(k), self.split(v)

#     # do scale dot product to compute similarity
#     out, attention = self.attention(q, k, v, mask, self.dropout)

#     out = self.concat(out)
#     out = self.w_concat(out)

#     return out # return multi head attention values

#   def split(self, tensor):
#         batch_size, length, d_model = tensor.size()

#         d_tensor = d_model // self.num_heads
#         tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)
#         # it is similar with group convolution (split by number of heads)

#         return tensor

#   def concat(self, tensor):
#         batch_size, head, length, d_tensor = tensor.size()
#         d_model = head * d_tensor

#         tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
#         return tensor

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
## now encoder 

class EncoderBlock(nn.Module):
    def __init__(self, features, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block= self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residuals_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    def forward(self, x, src_mask):
        x = self.residuals_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residuals_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features, self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residuals_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residuals_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residuals_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residuals_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:

    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encoding layers

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create the encoder and decoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
