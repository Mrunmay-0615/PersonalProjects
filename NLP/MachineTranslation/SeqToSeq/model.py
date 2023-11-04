import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k # German to English dataset
from torchtext.data import Field, BucketIterator # preprocessing
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, translate_sentence, bleu
from tqdm import tqdm
print("Imports Done")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

spacy_ger = spacy.load('de_core_news_lg')
spacy_eng = spacy.load('en_core_web_lg')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='<eos>')
print('tokenizers setup')


train_data, val_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                  fields=(german, english))

german.build_vocab(train_data, max_size=50000, min_freq=2)
english.build_vocab(train_data, max_size=50000, min_freq=2)
print("Data obtained and vocabularies built")


class Encoder(nn.Module):
    
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
    
    def forward(self, x):
        # x shape (Seq_len, N)
        embedding = self.dropout(self.embedding(x))
        # shape = (Seq_len, N, embed_size)
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers, dropout) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        # shape of x (N)
        # But we want (1, N)
        x = x.unsqueeze(0)
        embed = self.dropout(self.embedding(x))
        # shape = 1, N, embed_size

        outputs, (hidden, cell) = self.rnn(embed, (hidden, cell))
        # shape of output = (1, N, hidden_size)

        preds = self.fc(cell).squeeze(0)
        # shape of preds = (1, N, vocab_output_size)
        return preds, hidden, cell


class Translator(nn.Module):
    
    def __init__(self, input_size_encoder, input_size_decoder, output_size, embed_size, hidden_size, num_layers, dropout) -> None:
        super().__init__()
        self.encoder = Encoder(input_size_encoder, embed_size, hidden_size, num_layers, dropout) 
        self.decoder = Decoder(input_size_decoder, embed_size, hidden_size, output_size, num_layers, dropout)
    
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        # source = (tar_len, batch_size)
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        hidden, cell = self.encoder(source)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            # output = (N, eng_vocab_size)
            best_guess = output.argmax(1)

            x = target[t] if random.random < teacher_force_ratio else best_guess
        
        return outputs


# Training Hyperparaters
num_epochs = 50
lr_rate = 0.001
batch_size = 64

# Model Hyperparameters
load_model = False
device = torch.device(device)
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embed_size = 300
decoder_embed_size = 300
hidden_size = 1024
num_layers = 4
encoder_dropout = 0.5
decoder_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plots')
step = 0

train_iterator, val_iterator, test_iterator = BucketIterator.splits((train_data, val_data, test_data),
                                                                    batch_size=batch_size,
                                                                    sort_within_batch=True,
                                                                    sort_key=lambda x:len(x.src),
                                                                    device=device)


model = Translator(input_size_encoder, input_size_decoder, output_size, encoder_embed_size,
                   hidden_size, num_layers, encoder_dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))


print('Training starts')

for epoch in range(num_epochs):
    loop = tqdm(train_iterator, ncols=150, desc=f'Epoch {epoch+1}')
    total_loss = 0

    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."
    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()

    checkpoint = {'state_dict': model.state_dict, 'optimizer':optimizer.state_dict}
    save_checkpoint(checkpoint)

    for batch_idx, batch in enumerate(loop):
        input_sen = batch.src.to(device)
        target_sen = batch.trg.to(device)
        output = model(input_sen, target_sen)
        # shape = (trg_len, batch_size, output_size)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        # To avoid exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('Training Loss', loss, global_step=step)
        loop.set_postfix({'loss' : total_loss / (batch_idx+1)})


bleu_score = bleu(test_data, model, german, english, device)
print(f"Bleu Score {bleu_score * 100 : .2f}")



