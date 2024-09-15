from transformers import T5Tokenizer, T5ForConditionalGeneration

def initialize_model_and_tokenizer(model_name='t5-base'):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {str(e)}")
        raise

def tokenize_function(examples, tokenizer):
    inputs = [f"context: {c} question: {q}" for c, q in zip(examples['context'], examples['question'])]
    targets = examples['answer']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
