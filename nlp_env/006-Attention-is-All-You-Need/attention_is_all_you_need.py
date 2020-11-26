#!/usr/bin/env python3

import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter("ignore", UserWarning)
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

from model import Encoder, Decoder, Seq2Seq

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train(model, iterator, optimizer, criterion, clip, scheduler=None):
    if scheduler:
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output, _ = model(src, trg[:, :-1])
            # output.shape = (batch_size, trg_len-1, out_dim)
            # trg.shape = (batch_size, trg_len)

            out_dim = output.shape[-1]
            output = output.contiguous().view(-1, out_dim)
            trg = trg[:,1:].contiguous().view(-1)
            # output.shape = (batch_size*(trg_len-1), out_dim)
            # trg.shape = (batch_size*(trg_len-1))
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

            break

        return epoch_loss

    else:
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output, _ = model(src, trg[:, :-1])
            # output.shape = (batch_size, trg_len-1, out_dim)
            # trg.shape = (batch_size, trg_len)

            out_dim = output.shape[-1]
            output = output.contiguous().view(-1, out_dim)
            trg = trg[:,1:].contiguous().view(-1)
            # output.shape = (batch_size*(trg_len-1), out_dim)
            # trg.shape = (batch_size*(trg_len-1))
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

        output, _ = model(src, trg[:, :-1])
        # output.shape = (batch_size, trg_len-1, out_dim)
        # trg.shape = (batch_size, trg_len)

        out_dim = output.shape[-1]
        output = output.contiguous().view(-1, out_dim)
        trg = trg[:,1:].contiguous().view(-1)
        # output.shape = (batch_size*(trg_len-1), out_dim)
        # trg.shape = (batch_size*(trg_len-1))
        loss = criterion(output, trg)
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_t, end_t):
    t = end_t - start_t
    m = int(t / 60)
    s = int(t - m*60)
    return m, s

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

def plot(history, skip_start=10, skip_end=5, log_lr=True, show_lr=None, ax=None,suggest_lr=True):
    lrs = history["lr"][skip_start:-skip_end]
    losses = history["loss"][skip_start:-skip_end]

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(lrs, losses)

    if suggest_lr:
        print("LR suggestion: steepest gradient")
        min_grad_idx = None
        min_grad_idx = (np.gradient(np.array(losses))).argmin()
        if min_grad_idx is not None:
            print("Suggested LR: {:.2E}".format(lrs[min_grad_idx]))
            ax.scatter(
                lrs[min_grad_idx],
                losses[min_grad_idx],
                s=75,
                marker="o",
                color="red",
                zorder=3,
                label="steepest gradient",
            )
            ax.legend()
    if log_lr:
        ax.set_xscale("log")

    ax.set_xlabel("lr")
    ax.set_ylabel("loss")

    plt.show()


if __name__ == "__main__":

    SEED = 1234
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    
    SRC = Field(tokenize = tokenize_de,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True,
                batch_first = True)
    
    TRG = Field(tokenize = tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True,
                batch_first = True)
    
    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                        fields = (SRC, TRG))
    
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 128
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
         batch_size = BATCH_SIZE,
         device = device)

    IN_DIM = len(SRC.vocab)
    OUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(IN_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    model.apply(initialize_weights)

    LR = 5e-4
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    writer = SummaryWriter()

    # mode = 'lr_finder'
    mode = 'train'
    # mode = 'test'

    if mode == 'lr_finder':
        optimizer = optim.Adam(model.parameters(), lr=1e-6)

        # history for lr_finder
        history_lr_finder = {"lr": [], "loss": []}
        best_loss_lr_finder = None
        lr_scheduler = ExponentialLR(optimizer, end_lr=100, num_iter=100)

        N_EPOCHS = 150
        CLIP = 1
        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss = train(model, train_iterator, optimizer, criterion, CLIP, lr_scheduler)
            valid_loss = evaluate(model, valid_iterator, criterion)
            end_time = time.time()
            m, s = epoch_time(start_time, end_time)

            history_lr_finder["lr"].append(lr_scheduler.get_lr()[0])
            lr_scheduler.step()
            if epoch == 0:
                best_loss_lr_finder = valid_loss
            else:
                smooth_f = 0.05
                valid_loss = smooth_f*valid_loss+(1-smooth_f)*history_lr_finder["loss"][-1]
                if valid_loss < best_loss_lr_finder:
                    best_loss_lr_finder = valid_loss

            history_lr_finder["loss"].append(valid_loss)
            if valid_loss > 5 * best_loss_lr_finder:
                break

            print(f'Epoch: {epoch+1:02} | Time: {m}m {s}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            # print(f'\t Best Loss: {best_valid_loss:.3f} |  Val. PPL: {math.exp(best_valid_loss):7.3f}')

        plot(history_lr_finder)

    elif mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        N_EPOCHS = 15
        CLIP = 1
        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)
            end_time = time.time()
            m, s = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut6-model.pt')
            print(f'Epoch: {epoch+1:02} | Time: {m}m {s}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Best Loss: {best_valid_loss:.3f} |  Val. PPL: {math.exp(best_valid_loss):7.3f}')

            writer.add_scalars('Loss', {'train loss': train_loss,
                                        'valid loss': valid_loss,
                                        'best loss': best_valid_loss}, epoch)
            writer.add_scalars('PPL Loss', {'train loss': math.exp(train_loss),
                                            'valid loss': math.exp(valid_loss),
                                            'best loss': math.exp(best_valid_loss)}, epoch)
        writer.flush()
        writer.close()

    elif mode == 'test':
        model.load_state_dict(torch.load('tut6-model.pt'))
        test_loss = evaluate(model, test_iterator, criterion)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    else:
        raise ValueError
