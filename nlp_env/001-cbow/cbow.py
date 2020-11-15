from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim


corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s a",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

def w2v(corpus, skip_window, method):
    # convert sentences into words
    words = [line.split(" ") for line in corpus]
    words = list(chain(*words))
    words = np.array(words)

    # get unique words and frequency of each words
    vocab, v_cnt = np.unique(words, return_counts=True)

    # sort vocab by frequency, from most to least
    vocab = vocab[np.argsort(v_cnt)[::-1]]

    v2i = {v: i for i, v in enumerate(vocab)}
    i2v = {i: v for i, v in enumerate(vocab)}

    js = [i for i in range(-skip_window, skip_window+1) if i != 0]
    pairs = []

    for c in corpus:
        words = c.split(" ")
        w_idx = [v2i[w] for w in words]
        if method == "skip_gram":
            pass
        elif method == "cbow":
            for i in range(skip_window, len(w_idx)-skip_window):
                context = []
                for j in js:
                    context.append(w_idx[i+j])
                pairs.append(context+[w_idx[i]])
        else:
            raise ValueError
    pairs = np.array(pairs)
    
    if method=="skip_gram":
        pass
    elif method=="cbow":
        x, y = pairs[:, :-1], pairs[:, -1:]
    else:
        raise ValueError
    return Dataset(x, y, v2i, i2v)

class Dataset:
    def __init__(self, x, y, v2i, i2v):
        self.x = x
        self.y = y
        self.v2i = v2i
        self.i2v = i2v
        self.vocab = v2i.keys()

    def sample(self, n):
        idx = np.random.randint(0, len(self.x), n)
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.v2i)
        

class CBOW(nn.Module):
    def __init__(self, v_dim, emb_dim, window):
        super(CBOW, self).__init__()
        self.v_dim = v_dim
        self.emb_dim = emb_dim
        self.window = window
        self.embedding = nn.Embedding(v_dim, emb_dim)
        self.fc = nn.Linear(window*2*emb_dim, emb_dim)

    def forward(self, x, y):
        embedded_x = self.embedding(x)
        out1 = torch.mean(embedded_x, 1, True).flatten(1)
        out2 = self.embedding(y).view(-1, self.emb_dim)
        return out1, out2

def train(model, data):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for t in range(50000):
        x, y = data.sample(16)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        out1, out2 = model(x, y)
        loss = criterion(out1, out2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (t+1)%200 == 0:
            print(f"step: {t} | loss: {loss.item():.6f}")

    print(model.embedding.weight.data)
    print(model.embedding.weight.data.shape)

    show_w2v_word_embedding(model, data, "cbow.png")

def show_w2v_word_embedding(model, data, path):
    word_emb = model.embedding.weight.data
    for i in range(len(data)):
        c = "blue"
        try:
            int(data.i2v[i])
        except ValueError:
            c = "red"
        plt.text(word_emb[i, 0], word_emb[i, 1], s=data.i2v[i], color=c, weight="bold")
    plt.xlim(word_emb[:, 0].min(), word_emb[:, 0].max())
    plt.ylim(word_emb[:, 1].min(), word_emb[:, 1].max())
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("embedding dim1")
    plt.ylabel("embedding dim2")
    plt.savefig(path, dpi=300, format="png")
    plt.show()

if __name__ == "__main__":
    d = w2v(corpus, skip_window=2, method="cbow")
    m = CBOW(v_dim=len(d), emb_dim=2, window=2)
    train(m, d)

