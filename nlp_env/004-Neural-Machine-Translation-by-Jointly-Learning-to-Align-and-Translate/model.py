import random
import torch
from torch import nn, optim

class Encoder(nn.Module):
    def __init__(self, in_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.embedding = nn.Embedding(in_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(2*enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape = (x_len, batch_size)
        embedded = self.dropout(self.embedding(x))
        # embedded.shape = (x_len, batch_size, emb_dim)
        output, h = self.gru(embedded)
        # output.shape = (x_len, batch_size, enc_hid_dim * n_direction)
        # h.shape = (n_layer * n_direction, batch_size, enc_hid_dim)

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN

        h = torch.tanh(self.fc(torch.cat((h[-2,:,:],h[-1,:,:]), dim=1)))
        # h.shape = (batch_size, dec_hid_dim)
        # output.shape = (x_len, batch_size, enc_hid_dim * n_direction)
        return output, h

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim*2+dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, enc_output):
        # h.shape = (batch_size, dec_hid_dim)
        # enc_output.shape = (x_len, batch_size, enc_hid_dim*n_direction)

        x_len = enc_output.shape[0]
        h = h.unsqueeze(1).repeat(1, x_len, 1)
        # h.shape = (batch_size, x_len, dec_hid_dim)
        enc_output = enc_output.permute(1, 0, 2)
        # enc_output.shape = (batch_size, x_len, enc_hid_dim*n_direction)
        energy = torch.tanh(self.attn(torch.cat((h, enc_output), dim=2)))
        # energy.shape = (batch_size, x_len, dec_hid_dim)
        attention = self.v(energy).squeeze(2)
        # attention.shape = (batch_size, x_len)
        return self.softmax(attention)

class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.dec_hid_dim = dec_hid_dim
        self.attention = attention
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.gru = nn.GRU(enc_hid_dim*2+emb_dim, dec_hid_dim)
        self.fc = nn.Linear(enc_hid_dim*2+dec_hid_dim+emb_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, enc_output):
        # x.shape = (batch_size)
        # h.shape = (batch_size, dec_hid_dim)
        # enc_output.shape = (x_len, batch_size, enc_hid_dim * n_directio)
        x = x.unsqueeze(0)
        # x.shape = (1, batch_size)
        embedded = self.dropout(self.embedding(x))
        # embedded.shape = (1, batch_size, emb_dim)

        a = self.attention(h, enc_output).unsqueeze(1)
        # a.shape = (batch_size, 1, x_len)

        enc_output = enc_output.permute(1, 0, 2)
        # enc_output.shape = (batch_size, x_len, enc_hid_dim * n_directio)

        weighted = torch.bmm(a, enc_output)
        # weighted.shape = (batch_size, 1, enc_hid_dim*n_direction)
        
        weighted = weighted.permute(1, 0, 2)
        # weighted.shape = (1, batch_size, enc_hid_dim*n_direction)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input.shape = (1, batch_size, emb_dim+enc_hid_dim*n_direction)

        output, h = self.gru(rnn_input, h.unsqueeze(0))
        # output.shape = (1, batch_size, dec_hid_dim)
        # h.shape = (n_layer*n_direction, batch_size, dec_hid_dim)

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == h).all()

        prediction = self.fc(torch.cat((output, weighted, embedded), dim=2).squeeze(0))
        # prediction.shape = (batch_size, out_dim)
        return prediction, h.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, tf_ratio=0.5):
        # src.shape = (src_len, batch_size)
        # trg.shape = (trg_len, batch_size)

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        
        out_buf = torch.zeros((trg_len, batch_size, self.decoder.out_dim),
                               device=self.device)

        enc_output, h = self.encoder(src)

        trg_in = trg[0, :]

        for t in range(1, trg_len):
            out, h = self.decoder(trg_in, h, enc_output)
            # out.shape = (batch_size, out_dim)
            out_buf[t] = out
            tf = random.random() < tf_ratio
            top1 = out.argmax(1)
            trg_in = trg[t] if tf else top1
        return out_buf

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

