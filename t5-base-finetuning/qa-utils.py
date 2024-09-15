import torch

def answer_question(question, context, model, tokenizer, max_len=64):
    input_text = f"context: {context} question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            num_beams=4,
            no_repeat_ngram_size=2,
            min_length=1,
            max_length=max_len,
            early_stopping=True
        )
    
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return answer

def interactive_qa(model, tokenizer):
    print("Welcome to the interactive Question Answering system!")
    print("Enter 'quit' to exit.")
    
    while True:
        context = input("\nEnter the context (or 'quit' to exit): ")
        if context.lower() == 'quit':
            break
        
        question = input("Enter your question: ")
        if question.lower() == 'quit':
            break
        
        answer = answer_question(question, context, model, tokenizer)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
