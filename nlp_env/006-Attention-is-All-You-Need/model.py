import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layer, n_head,
                 pf_dim, dropout, device, max_length=100):
        super(Encoder, self).__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(in_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_head,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor([hid_dim],
                                             dtype=torch.float32,
                                             device=device))

    def forward(self, src, src_mask):
        # src.shape = (batch_size, src_len)
        # src_mask.shape = (batch_size, src_len)
        batch_size, src_len = src.shape
        pos = torch.arange(0, src_len, device=self.device)
        pos = pos.unsqueeze(0).repeat(batch_size, 1)
        # pos.shape = (batch_size, src_len)
        src = self.dropout(self.tok_embedding(src)*self.scale +
                           self.pos_embedding(pos))
        # src.shape = (batch_size, src_len, hid_dim)

        for layer in self.layers:
            src = layer(src, src_mask)
        # src.shape = (batch_size, src_len, hid_dim)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_head, pf_dim, dropout, device):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_head,
                                                      dropout, device)
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src.shape = (batch_size, src_len, hid_dim)
        # src_mask.shape = (batch_size, src_len)

        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.layernorm1(src+self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.layernorm2(src+self.dropout(_src))

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_head, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()
        assert hid_dim % n_head == 0

        self.hid_dim = hid_dim
        self.n_head = n_head
        self.head_dim = hid_dim // n_head

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.tensor([hid_dim],
                                             dtype=torch.float32,
                                             device=device))

    def forward(self, q, k, v, mask=None):
        # q_len == k_len == v_len
        # q.shape = (batch_size, q_len, hid_dim)
        # k.shape = (batch_size, k_len, hid_dim)
        # v.shape = (batch_size, v_len, hid_dim)
        batch_size = q.shape[0]

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        # q.shape = (batch_size, q_len, hid_dim)
        # k.shape = (batch_size, k_len, hid_dim)
        # v.shape = (batch_size, v_len, hid_dim)

        q = q.view(batch_size, -1, self.n_head, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_head, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_head, self.head_dim)
        v = v.permute(0, 2, 1, 3)
        # q.shape = (batch_size, n_head, q_len, head_dim)
        # k.shape = (batch_size, n_head, k_len, head_dim)
        # v.shape = (batch_size, n_head, v_len, head_dim)

        """
        Scaled Dot-Product Attention
            Attention(q, k, v) = softmax(q * k^T / sqrt(hid_dim)) * v
        """
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # energy.shape = (batch_size, n_head, q_len, k_len)

        attention = torch.softmax(energy, dim=-1)
        # attention.shape = (batch_size, n_head, q_len, k_len)

        x = torch.matmul(self.dropout(attention), v)
        # x.shape = (batch_size, n_head, q_len, head_dim)
        """
        Scaled Dot-Product Attention end
        """

        x = x.permute(0, 2, 1, 3).contiguous()
        # x.shape = (batch_size, q_len, n_head, head_dim)

        x = x.view(batch_size, -1, self.hid_dim)
        # x.shape = (batch_size, q_len, hid_head)

        x = self.fc_o(x)
        # x.shape = (batch_size, q_len, hid_head)

        return x, attention


class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionwiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, hid_dim)
        x = self.dropout(torch.relu(self.fc1(x)))
        # x.shape = (batch_size, seq_len, pf_dim)
        x = self.fc2(x)
        # x.shape = (batch_size, seq_len, hid_dim)
        return x


class Decoder(nn.Module):
    def __init__(self, out_dim, hid_dim, n_layer, n_head,
                 pf_dim, dropout, device, max_length=100):
        super(Decoder, self).__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(out_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_head,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layer)])
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor([hid_dim],
                                dtype=torch.float32,
                                device=device))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg.shape = (batch_size, trg_len)
        # enc_src.shape = (batch_size, src_len, hid_dim)
        # trg_mask = (batch_size, trg_len)
        # src_mask = (batch_size, src_len)

        batch_size, trg_len = trg.shape
        pos = torch.arange(0, trg_len, device=self.device)
        pos = pos.unsqueeze(0).repeat(batch_size, 1)
        # pos.shape = (batch_size, trg_len)

        trg = self.dropout(self.tok_embedding(trg)*self.scale +
                           self.pos_embedding(pos))
        # trg.shape = (batch_size, trg_len, hid_dim)

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg.shape = (batch_size, trg_len, hid_dim)

        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_head, pf_dim, dropout, device):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_head,
                                                      dropout, device)
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.enc_attention = MultiHeadAttentionLayer(hid_dim, n_head,
                                                     dropout, device)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.layernorm3 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg.shape = (batch_size, trg_len)
        # enc_src = (batch_size, src_len, hid_dim)
        # trg_mask = (batch_size, trg_len)
        # src_mask = (batch_size, src_len)

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.layernorm1(trg + self.dropout(_trg))
        # trg.shape = (batch_size, trg_len, hid_dim)
        _trg, attention = self.enc_attention(trg, enc_src, enc_src, src_mask)
        trg = self.layernorm2(trg + self.dropout(_trg))
        # trg.shape = (batch_size, trg_len, hid_dim)
        _trg = self.positionwise_feedforward(trg)
        trg = self.layernorm3(trg + self.dropout(_trg))
        # trg.shape = (batch_size, trg_len, hid_dim)
        # attention.shape = (batch_size, n_head, trg_len, src_len)
        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, src_pad_idx, trg_pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.enc = enc
        self.dec = dec
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src.shape = (batch_size, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(1)
        # src_mask.shape = (batch_size, 1, 1, src_len)
        return src_mask

    def make_trg_mask(self, trg):
        # trg.shape = (batch_size, trg_len)
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(1)
        # trg_mask.shape = (batch_size, 1, 1, trg_len)

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len),
                                             device=self.device)).bool()
        # trg_sub_mask.shape = (trg_len, trg_len)
        trg_mask = trg_mask & trg_sub_mask
        # trg_mask.shape = (batch_size, 1, trg_len, trg_len)
        return trg_mask

    def forward(self, src, trg):
        # src.shape = (batch_size, src_len)
        # trg.shape = (batch_size, trg_len)

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask.shape = (batch_size, 1, 1, src_len)
        # trg_mask.shape = (batch_size, 1, trg_len, trg_len)

        enc_src = self.enc(src, src_mask)
        # enc_src.shape = (batch_size, src_len, hid_dim)
        output, attention = self.dec(trg, enc_src, trg_mask, src_mask)
        # output.shape = (batch_size, trg_len, out_dim)
        # attention.shape = (batch_size, n_head, trg_len, src_len)
        return output, attention
