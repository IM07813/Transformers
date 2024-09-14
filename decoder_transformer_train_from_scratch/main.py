import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from datasets import load_dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu

from model import DecoderOnlyTransformer
from data_processing import create_vocab, create_dataloaders

# device if gpu is available else cpu. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
seq_length = 128
emsize = 768
d_hid = 3072
nlayers = 12
nhead = 12
dropout = 0.2
epochs = 50

# Training function
def train(model, train_loader, val_loader, vocab_size, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_loss = float('inf')
    best_model = None
    patience = 3
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for batch, (data, targets) in progress_bar:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch + 1)})

        val_loss = evaluate(model, val_loader, criterion, vocab_size)
        perplexity = math.exp(val_loss)
        print(f'| End of epoch {epoch:3d} | valid loss {val_loss:5.2f} | perplexity {perplexity:8.2f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model)
    return model

def evaluate(model, data_loader, criterion, vocab_size):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            total_loss += criterion(output.view(-1, vocab_size), targets.view(-1)).item() * data.size(0)
    return total_loss / len(data_loader.dataset)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_text(model, vocab, input_text, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    words = input_text.split()
    input_ids = torch.tensor([[vocab.get(w, vocab['<unk>']) for w in words]], dtype=torch.long).to(device)
    
    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_ids)
        
        next_token_logits = output[0, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        next_word = list(vocab.keys())[list(vocab.values()).index(next_token.item())]
        words.append(next_word)
        
        if next_word == '<unk>':
            break
    
    return ' '.join(words)

def calculate_perplexity(model, data_loader, vocab_size):
    model.eval()
    total_loss = 0.0
    total_words = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
            total_words += targets.numel()
    return math.exp(total_loss / total_words)

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def main():
    print("Loading datasets...")
    try:
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
        c4 = load_dataset("c4", "en", split="train", streaming=True)
        c4 = list(c4.take(1000000))  
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print("Creating vocabulary...")
    vocab, vocab_size = create_vocab([wikitext['train'], c4], max_vocab_size=50000)

    print("Preparing dataloaders...")
    try:
        train_loader = create_dataloaders(wikitext['train'], vocab, batch_size, seq_length)
        val_loader = create_dataloaders(wikitext['validation'], vocab, batch_size, seq_length)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    print("Initializing the model...")
    model = DecoderOnlyTransformer(vocab_size, emsize, nhead, d_hid, nlayers, dropout).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    print("Starting training...")
    model = train(model, train_loader, val_loader, vocab_size, epochs)

    print("Saving the model...")
    torch.save(model.state_dict(), 'decoder_only_transformer.pth')

    print("Training completed and model saved.")

    print("Evaluating the model...")
    perplexity = calculate_perplexity(model, val_loader, vocab_size)
    print(f"Final Perplexity: {perplexity:.2f}")

    print("Generating text...")
    input_text = "The quick brown fox"
    generated_text = generate_text(model, vocab, input_text)
    print(f"Generated text: {generated_text}")

    print("Calculating BLEU score...")
    reference = "The quick brown fox jumps over the lazy dog"
    bleu_score = calculate_bleu(reference, generated_text)
    print(f"BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    main()
