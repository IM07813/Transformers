# Transformer Language Model (Decoder-Only)

This project implements a Transformer Language Model from scratch using only NumPy. It's a simplified version of the decoder-only transformer architecture, similar to GPT models.

## Requirements: 
numpy==1.21.0

## Overview

The Transformer Language Model is composed of the following main components:

1. Positional Encoding
2. Multi-Head Attention
3. Feed-Forward Neural Network
4. Layer Normalization
5. Decoder Block
6. Full Decoder
7. Transformer Language Model

Each component is implemented in a separate Python file for modularity and ease of understanding.

## File Structure

- `positional_encoding.py`: Implements the positional encoding layer.
- `attention.py`: Contains the Multi-Head Attention mechanism.
- `feed_forward.py`: Implements the Feed-Forward Neural Network.
- `layer_norm.py`: Contains Layer Normalization and residual connection functions.
- `decoder.py`: Implements the Decoder Block and full Decoder.
- `transformer_lm.py`: Combines all components into the Transformer Language Model.
- `main.py`: Demonstrates how to use the Transformer Language Model.

## How It Works

1. **Positional Encoding**: Adds positional information to the input embeddings, allowing the model to understand the sequence order.

2. **Multi-Head Attention**: Performs self-attention on the input, allowing the model to weigh the importance of different parts of the input when processing each token.

3. **Feed-Forward Neural Network**: Applies a simple neural network to each position separately and identically.

4. **Layer Normalization**: Normalizes the outputs of the attention and feed-forward layers, helping with training stability.

5. **Decoder Block**: Combines the above components with residual connections.

6. **Full Decoder**: Stacks multiple Decoder Blocks and applies an initial embedding layer.

7. **Transformer Language Model**: Wraps the Decoder and adds a final output layer to produce logits for the vocabulary.

## Implementation Details

- The entire model is implemented using only NumPy, without any deep learning frameworks.
- The model uses the GELU activation function in the Feed-Forward Neural Network.
- Dropout is applied in various parts of the model to prevent overfitting.
- The model initializes weights using the Xavier/Glorot initialization method.

## Usage

To run the model:

1. Ensure you have NumPy installed (see requirements.txt).
2. Run the `main.py` file:

```
python main.py
```

This will create a Transformer Language Model instance, generate random input data, run it through the model, and display the output shape and a sample of the output logits.

## Limitations

- This implementation is for educational purposes and is not optimized for large-scale language modeling tasks.
- The model doesn't include training code or more advanced features like mixed-precision training or parallelism.
- Performance may be significantly slower compared to optimized implementations using deep learning frameworks.

## Future Improvements

- Implement training logic with backpropagation.
- Add support for loading and saving model weights.
- Optimize for better performance, possibly by using libraries like JAX or PyTorch.
- Implement more advanced features like adaptive attention span or sparse attention.

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." arXiv:1706.03762.
- Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training."
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."

