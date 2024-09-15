import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    
    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        return np.matmul(attention_weights, V)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def dropout(self, x):
        mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return x * mask
    
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.shape
        
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        return self.dropout(np.matmul(attention_output, self.W_o))
