import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
        self.dropout_rate = dropout_rate
    
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def dropout(self, x):
        mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return x * mask
    
    def forward(self, x):
        x = self.gelu(np.matmul(x, self.W1) + self.b1)
        x = self.dropout(x)
        return np.matmul(x, self.W2) + self.b2
