import numpy as np
from transformer import TransformerLM

def main():
    # Hyperparameters
    vocab_size = 10000
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    max_seq_len = 1024
    dropout_rate = 0.1

    # Create a TransformerLM instance
    model = TransformerLM(vocab_size, num_layers, d_model, num_heads, d_ff, max_seq_len, dropout_rate)

    # Generate random input data
    batch_size = 4
    seq_len = 50
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Run the data through the model
    output = model.forward(input_ids)

    print("Output shape:", output.shape)
    print("Output example (first sequence, first 10 logits):")
    print(output[0, 0, :10])

if __name__ == "__main__":
    main()
