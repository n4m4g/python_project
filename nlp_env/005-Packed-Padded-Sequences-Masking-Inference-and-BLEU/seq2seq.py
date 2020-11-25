#!/usr/bin/env python3
import random
import math
import time
from copy import deepcopy

import spacy
import numpy as np

# viewing the attention
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter("ignore", UserWarning)
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from model import init_weights
from model import Encoder, Attention, Decoder, Seq2Seq


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip, scheduler=None):
    if scheduler:
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output = model(src, src_len, trg)

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

            break
        
        mean_loss = epoch_loss / 1
        return mean_loss

    else:
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output = model(src, src_len, trg)

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
        
        mean_loss = epoch_loss / len(iterator)
        return mean_loss

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        trg = batch.trg

        output = model(src, src_len, trg, 0)

        # trg.shape = (trg_len, batch_size)
        # output.shape = (trg_len, batch_size, out_dim)
        trg = trg[1:].view(-1)
        output = output[1:].view(-1, output.shape[-1])
        # trg.shape = ((trg_len-1) * batch_size)
        # output.shape = ((trg_len-1) * batch_size, out_dim)

        loss = criterion(output, trg)
        epoch_loss += loss.item()

    mean_loss = epoch_loss / len(iterator)
    return mean_loss

def epoch_time(start_t, end_t):
    t = end_t - start_t
    m = int(t / 60)
    s = int(t - m*60)
    return m, s

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # convert string to index
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # convert list into LongTensor
    src = torch.tensor(src_indexes, dtype=torch.long, device=device).unsqueeze(1)
    # src.shape = (src_len, 1)
    src_len = torch.tensor([len(src_indexes)], dtype=torch.long, device=device)
    # src_len.shape = (src_len,)

    with torch.no_grad():
        enc_output, h = model.encoder(src, src_len)
        # enc_output.shape = (src_len, 1, enc_dim*n_direction)
        # h.shape = (batch_size, dec_hid_dim)

    mask = model.create_mask(src)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros((max_len, 1, len(src_indexes)), device=device)

    for i in range(max_len):
        trg = torch.tensor([trg_indexes[-1]], dtype=torch.long, device=device)

        with torch.no_grad():
            output, h, attentions[i] = model.decoder(trg, h, enc_output, mask)

        pred_token = output.argmax(dim=1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[index] for index in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def display_attention(sentence, translation, attention):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()
    # attention.shape = (max_len, src_len)

    cax = ax.matshow(np.flip(attention, axis=1), cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence][::-1]+['<eos>'],
                       rotation=45)
    ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

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

    SEED=1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    """
    python3 -m spacy download de
    python3 -m spacy download en
    """
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    """
    adding a few improvements
        Packed padded sequences are used to tell our RNN to skip over padding tokens in our encoder.
        Masking explicitly forces the model to ignore certain values, such as attention over padded elements.
    """

    """
    include_lengths
        cause our batch.src to be a tuple. 
        The first element of the tuple is the same as before,
        a batch of numericalized source sentence as a tensor,
        the second element is the non-padded lengths of each source sentence within the batch.
    """
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                include_lengths=True)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    # https://torchtext.readthedocs.io/en/latest/datasets.html#multi30k
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

    BATCH_SIZE = 300
    
    """
    packed padded sequences is that all elements in the batch
    need to be sorted by their non-padded lengths in descending order
    the first sentence in the batch needs to be the longest.

    sort_within_batch
        tells the iterator that the contents of the batch need to be sorted
    sort_key
        tells the iterator how to sort the elements in the batch.
        Here, we sort by the length of the src sentence.
    """
    train_iter, valid_iter, test_iter = BucketIterator.splits(
                                            (train_data, valid_data, test_data),
                                            batch_size=BATCH_SIZE,
                                            device=device,
                                            sort_within_batch=True,
                                            sort_key=lambda x: len(x.src))

    IN_DIM = len(SRC.vocab)
    OUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    SRC_PAD_INDEX = SRC.vocab.stoi[SRC.pad_token]

    enc = Encoder(IN_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    dec = Decoder(OUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, SRC_PAD_INDEX, device).to(device)
    model.apply(init_weights)

    print(f"Model has {count_parameters(model):,} trainable parameters")

    # TRG.pad_token = <pad>
    # TRG_PAD_IDX = 1
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # tensorboard writer
    writer = SummaryWriter()

    # mode = 'lr_finder'
    mode = 'train'
    # mode = 'eval'

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
            start_t = time.time()
            train_loss = train(model, train_iter, optimizer, criterion, CLIP, lr_scheduler)
            valid_loss = evaluate(model, valid_iter, criterion)
            end_t = time.time()
            m, s = epoch_time(start_t, end_t)

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
            print(f'\t Best Loss: {best_valid_loss:.3f} |  Best PPL: {math.exp(best_valid_loss):7.3f}')

        plot(history_lr_finder)

    elif mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=0.000811)

        N_EPOCHS = 20
        CLIP = 1

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_t = time.time()
            train_loss = train(model, train_iter, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iter, criterion)
            end_t = time.time()
            m, s = epoch_time(start_t, end_t)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut1-model.pt')

            print(f'Epoch: {epoch+1:02} | Time: {m}m {s}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Best Loss: {best_valid_loss:.3f} |  Best PPL: {math.exp(best_valid_loss):7.3f}')

            writer.add_scalars('Loss', {'train loss': train_loss,
                                        'valid loss': valid_loss,
                                        'best loss': best_valid_loss}, epoch)
            writer.add_scalars('PPL Loss', {'train loss': math.exp(train_loss),
                                            'valid loss': math.exp(valid_loss),
                                            'best loss': math.exp(best_valid_loss)}, epoch)
        writer.flush()
        writer.close()


    elif mode == 'eval':
        model.load_state_dict(torch.load('tut1-model-400.pt'))
        test_loss = evaluate(model, test_iter, criterion)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        # train_data
        example_idx = 12

        src = vars(train_data.examples[example_idx])['src']
        trg = vars(train_data.examples[example_idx])['trg']
        
        print(f'src = {src}')
        print(f'trg = {trg}')

        translation, attention = translate_sentence(src, SRC, TRG, model, device)

        print(f'predicted trg = {translation}')

        display_attention(src, translation, attention)


    else:
        assert False, 'wrong mode'

    
