import numpy as np
import matplotlib.pyplot as plt
import spacy
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    # text : sentence in string
    spacy_de = spacy.load('de')
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    # text : sentence in string
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def epoch_time(start_t, end_t):
    t = end_t - start_t
    m = int(t / 60)
    s = int(t - m*60)
    return m, s


def plot(history, skip_start=10, skip_end=5, log_lr=True,
         show_lr=None, ax=None, suggest_lr=True):
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


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r
                for base_lr in self.base_lrs]
