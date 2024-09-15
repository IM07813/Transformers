import numpy as np
from decoder import Decoder

class TransformerLM:
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, max_seq_len=5000, dropout_rate=0.1):
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, max_seq_len, dropout_rate)
        self.final_layer = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)
    
    def forward(self, x):
        x = np.take(self.embedding, x, axis=0) * np.sqrt(self.d_model)
        x = self.decoder.forward(x)
        return np.matmul(x, self.final_layer)
