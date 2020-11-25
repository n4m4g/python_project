#!/usr/bin/env python3

import random
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np


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
