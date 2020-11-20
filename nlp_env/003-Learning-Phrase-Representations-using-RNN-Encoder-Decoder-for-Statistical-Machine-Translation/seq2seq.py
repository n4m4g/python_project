#!/usr/bin/env python3
import random
import math
import time
from copy import deepcopy

import spacy
import numpy as np
import torch
from torch import nn, optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from model import init_weights
from model import Encoder, Decoder, Seq2Seq

SEED=1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg)

        # trg.shape = (trg_len, batch_size)
        # output.shape = (trg_len, batch_size, out_dim)
        trg = trg[1:].view(-1)
        output = output[1:].view(-1, output.shape[-1])
        # trg.shape = ((trg_len-1) * batch_size)
        # output.shape = ((trg_len-1) * batch_size, out_dim)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        output = model(src, trg, 0)

        # trg.shape = (trg_len, batch_size)
        # output.shape = (trg_len, batch_size, out_dim)
        trg = trg[1:].view(-1)
        output = output[1:].view(-1, output.shape[-1])
        # trg.shape = ((trg_len-1) * batch_size)
        # output.shape = ((trg_len-1) * batch_size, out_dim)

        loss = criterion(output, trg)
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_t, end_t):
    t = end_t - start_t
    m = int(t / 60)
    s = int(t - m*60)
    return m, s

            

if __name__ == "__main__":
    """
    python3 -m spacy download en
    python3 -m spacy download de
    """
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG))

    print(f"# of training examples: {len(train_data.examples)}")
    print(f"# of validation examples: {len(valid_data.examples)}")
    print(f"# of test examples: {len(test_data.examples)}")
    print(vars(train_data.examples[0]))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    print(f"Unique tokens in source (de) vocab: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocab: {len(TRG.vocab)}")

    device = torch.device('cuda')

    BATCH_SIZE = 256
    train_iter, valid_iter, test_iter = BucketIterator.splits(
                                            (train_data, valid_data, test_data),
                                            batch_size=BATCH_SIZE,
                                            device=device)

    IN_DIM = len(SRC.vocab)
    OUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYER = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(IN_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(IN_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(f"Model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(model.parameters())

    # TRG.pad_token = <pad>
    # TRG_PAD_IDX = 1
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 20
    CLIP = 1

    best_valid_loss = float('inf')
    best_model_weights = deepcopy(model.state_dict())

    for epoch in range(N_EPOCHS):
        start_t = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)
        end_t = time.time()
        m, s = epoch_time(start_t, end_t)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_weights = deepcopy(model.state_dict())

        print(f'Epoch: {epoch+1:02} | Time: {m}m {s}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), 'tut1-model.pt')

    model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss = evaluate(model, test_iter, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    
