import json
from datasets import Dataset

def load_squad(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            squad_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found. Please check the path.")
    except json.JSONDecodeError:
        raise ValueError(f"The file {file_path} is not a valid JSON file.")

    contexts, questions, answers = [], [], []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer['text'])
    
    return Dataset.from_dict({
        'context': contexts,
        'question': questions,
        'answer': answers
    })
