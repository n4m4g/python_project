import random
import torch
from torch import nn, optim

class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim, hid_dim, dropout):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(in_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape = (x_len, batch_size)
        embedded = self.dropout(self.embedding(x))
        # embedded.shape = (x_len, batch_size, emb_dim)
        output, h = self.gru(embedded)
        # output.shape = (x_len, batch_size, hid_dim * n_direction)
        # h.shape = (n_layer * n_direction, batch_size, hid_dim)
        return h

class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, dropout):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.gru = nn.GRU(emb_dim+hid_dim, hid_dim)
        self.fc = nn.Linear(emb_dim+2*hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, context):
        # x.shape = (batch_size)
        # h.shape = (n_layer * n_direction, batch_size, hid_dim)
        x = x.unsqueeze(0)
        # x.shape = (1, batch_size)
        embedded = self.dropout(self.embedding(x))
        # embedded.shape = (1, batch_size, emb_dim)
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con.shape = (1, batch_size, emb_dim+hid_dim)
        output, h = self.gru(emb_con, h)
        # output.shape = (1, batch_size, hid_dim*n_direction)
        # h.shape = (n_layer * n_direction, batch_size, hid_dim)
        output_con = torch.cat((embedded, h, context), dim=2)
        # output_con.shape = (1, batch_size, emb_dim+2(hid_dim))
        prediction = self.fc(output_con.squeeze(0))
        # prediction.shape = (batch_size, out_dim)
        return prediction, h
        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim

    def forward(self, src, trg, tf_ratio=0.5):
        # src.shape = (src_len, batch_size)
        # trg.shape = (trg_len, batch_size)

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        
        out_buf = torch.zeros((trg_len, batch_size, self.decoder.out_dim),
                               device=self.device)

        h = self.encoder(src)
        context = h

        trg_in = trg[0, :]

        for t in range(1, trg_len):
            out, h = self.decoder(trg_in, h, context)
            # out.shape = (batch_size, out_dim)
            out_buf[t] = out
            tf = random.random() < tf_ratio
            top1 = out.argmax(1)
            trg_in = trg[t] if tf else top1
        return out_buf

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

