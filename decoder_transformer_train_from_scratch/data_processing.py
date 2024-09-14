import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

def create_vocab(datasets, max_vocab_size=50000):
    counter = Counter()
    for dataset in datasets:
        if isinstance(dataset, dict) and 'text' in dataset:
            for text in dataset['text']:
                counter.update(text.split())
        elif isinstance(dataset, list):
            for item in dataset:
                if isinstance(item, dict) and 'text' in item:
                    counter.update(item['text'].split())
        elif hasattr(dataset, 'features') and 'text' in dataset.features:
            for text in dataset['text']:
                counter.update(text.split())
    vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(max_vocab_size-2))}
    vocab['<unk>'] = 1
    vocab['<pad>'] = 0
    return vocab, len(vocab)

class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        words = text.split()
        tokens = [self.vocab.get(word, self.vocab['<unk>']) for word in words]

        if len(tokens) > self.seq_length:
            start_idx = random.randint(0, len(tokens) - self.seq_length - 1)
            tokens = tokens[start_idx:start_idx + self.seq_length + 1]
        else:
            tokens = tokens + [self.vocab['<pad>']] * (self.seq_length + 1 - len(tokens))

        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

def create_dataloaders(dataset, vocab, batch_size, seq_length):
    if isinstance(dataset, dict) and 'text' in dataset:
        texts = dataset['text']
    elif isinstance(dataset, list):
        texts = [item['text'] for item in dataset if isinstance(item, dict) and 'text' in item]
    elif hasattr(dataset, 'features') and 'text' in dataset.features:
        texts = dataset['text']
    else:
        raise ValueError(f"Unsupported dataset format: {type(dataset)}")
    
    dataset = TextDataset(texts, vocab, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
