import os
from data_loading import load_squad
from model_utils import initialize_model_and_tokenizer
from training import train_model
from qa_utils import interactive_qa, answer_question

def main():
    # Set up file paths
    train_file = 'data/train-v1.1.json'
    val_file = 'data/dev-v1.1.json'
    model_output_dir = './t5_squad_model'

    # if files exist or not 
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("Error: SQuAD dataset files not found. Please make sure the directory path is defined correctly.")
        return

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_squad(train_file)
    val_dataset = load_squad(val_file)

    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer()

    # Train the model
    print("Training the model...")
    train_model(model, tokenizer, train_dataset, val_dataset, model_output_dir)

    # Test the model
    print("\nTesting the model with a sample question:")
    context = "The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage."
    question = "What is SQuAD?"
    print("Question:", question)
    print("Answer:", answer_question(question, context, model, tokenizer))

    # Run interactive QA session
    print("\nStarting interactive QA session...")
    interactive_qa(model, tokenizer)

if __name__ == "__main__":
    main()
