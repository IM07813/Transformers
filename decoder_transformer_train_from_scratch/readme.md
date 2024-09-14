# Decoder-Only Transformer for Text Generation

This project implements a decoder-only transformer model for text generation using PyTorch. The model is trained on the WikiText-2 dataset and a subset of the C4 dataset.

## Project Structure

The project consists of three main Python files:

1. `main.py`: Contains the main execution logic, training loop, evaluation functions, and text generation.
2. `model.py`: Defines the PositionalEncoding and DecoderOnlyTransformer model classes.
3. `data_processing.py`: Includes functions for creating the vocabulary and data loaders.

## Requirements

This project requires Python 3.7+ and the following libraries:

- PyTorch
- Datasets
- Transformers
- tqdm
- NLTK

You can install the required libraries using the `requirements.txt` file.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/decoder-only-transformer.git
   cd decoder-only-transformer
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model and generate text, run:

```
python main.py
```

This script will:
1. Load the WikiText-2 and C4 datasets
2. Create a vocabulary
3. Initialize the transformer model
4. Train the model
5. Save the trained model
6. Evaluate the model's perplexity
7. Generate sample text
8. Calculate BLEU score for the generated text

## Model Architecture

The model is a decoder-only transformer with the following hyperparameters:

- Embedding size: 768
- Hidden size: 3072
- Number of attention heads: 12
- Number of layers: 12
- Dropout: 0.1

## Training

The model is trained for 10 epochs with early stopping based on validation loss. It uses AdamW optimizer with a learning rate of 5e-5 and a linear warmup schedule.

## Text Generation

Text generation uses top-k and top-p (nucleus) sampling to produce diverse and coherent text outputs.

## Evaluation

The model is evaluated using perplexity on the validation set and BLEU score for generated text.

## Notes

- This project is computationally intensive and may take a considerable amount of time to run, especially on CPU. Ensure you have sufficient computational resources available.
- The C4 dataset is limited to 1,000,000 examples to reduce memory usage and training time. You can modify this in the `main.py` file if you have more computational resources available.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The transformer model implementation is based on the PyTorch tutorial on sequence-to-sequence modeling.
- Datasets are provided by Hugging Face's Datasets library.
