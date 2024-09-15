import numpy as np
from positionalencoding import PositionalEncoding
from attention import MultiHeadAttention
from feedforward import FeedForward
from layernorm import LayerNorm, residual_connection

class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def forward(self, x, self_attn_mask=None):
        attn_output = self.self_attn.forward(x, x, x, self_attn_mask)
        x = residual_connection(x, attn_output, self.dropout_rate)
        x = self.norm1.forward(x)
        
        ffn_output = self.ffn.forward(x)
        x = residual_connection(x, ffn_output, self.dropout_rate)
        return self.norm2.forward(x)

class Decoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len=5000, dropout_rate=0.1):
        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = [DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.dropout_rate = dropout_rate
    
    def generate_look_ahead_mask(self, seq_len):
        mask = 1 - np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask[np.newaxis, np.newaxis, :, :]
    
    def forward(self, x, self_attn_mask=None):
        seq_len = x.shape[1]
        x = self.pos_encoder.forward(x)
        x = self.dropout(x)
        
        if self_attn_mask is None:
            self_attn_mask = self.generate_look_ahead_mask(seq_len)
        
        for layer in self.layers:
            x = layer.forward(x, self_attn_mask)
        
        return x
    
    def dropout(self, x):
        mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return x * mask
