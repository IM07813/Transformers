import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

def residual_connection(x, sublayer_output, dropout_rate=0.1):
    mask = (np.random.rand(*sublayer_output.shape) > dropout_rate) / (1 - dropout_rate)
    return x + sublayer_output * mask
