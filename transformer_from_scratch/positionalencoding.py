import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.positional_encoding = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.positional_encoding[:, 0::2] = np.sin(position * div_term)
        self.positional_encoding[:, 1::2] = np.cos(position * div_term)
    
    def forward(self, x):
        return x + self.positional_encoding[:x.shape[1]]
