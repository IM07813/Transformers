# T5 Finetuning for SQuAD Question Answering

This project demonstrates how to finetune a T5 model on the Stanford Question Answering Dataset (SQuAD) for question answering tasks. The code includes data loading, model training, and an interactive QA system.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/t5-squad-finetuning.git
   cd t5-squad-finetuning
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset

This project uses the Stanford Question Answering Dataset (SQuAD) v1.1. You need to download the dataset files and place them in the project root directory:

- Download [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- Download [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

## Project Structure

- `main.py`: The main script to run the entire pipeline
- `data_loading.py`: Functions for loading and preprocessing the SQuAD dataset
- `model_utils.py`: Utility functions for initializing the model and tokenizer
- `training.py`: Functions for training the model
- `qa_utils.py`: Utilities for question answering, including an interactive QA system

## Usage

To run the entire pipeline (data loading, model training, and interactive QA):

```
python main.py
```

This will:
1. Load the SQuAD dataset
2. Initialize the T5 model and tokenizer
3. Train the model on the SQuAD dataset
4. Save the trained model
5. Test the model with a sample question
6. Start an interactive QA session

## How It Works

1. **Data Loading**: The `load_squad` function in `data_loading.py` reads the SQuAD JSON files and converts them into a format suitable for training.

2. **Model Initialization**: The `initialize_model_and_tokenizer` function in `model_utils.py` loads the pretrained T5 model and tokenizer.

3. **Training**: The `train_model` function in `training.py` sets up the training arguments, tokenizes the dataset, and trains the model using the Hugging Face `Trainer` class.

4. **Question Answering**: The `answer_question` function in `qa_utils.py` takes a question and context, processes them through the model, and returns the generated answer.

5. **Interactive QA**: The `interactive_qa` function in `qa_utils.py` provides a command-line interface for users to input contexts and questions, and receive answers from the model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
