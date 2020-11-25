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
        pos = torch.arange(0, src_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        # pos.shape = (batch_size, src_len)
        src = self.dropout(self.tok_embedding(src)*self.scale+self.pos_embedding(pos))
        # src.shape = (batch_size, src_len, hid_dim)

        for layer in self.layers:
            src = layer(src, src_mask)
        # src.shape = (batch_size, src_len, hid_dim)

        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_head, pf_dim, dropout, device):
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_head, dropout, device)
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hid_dim, pf_dim, dropout)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src.shape = (batch_size, src_len, hid_dim)
        # src_mask.shape = (batch_size, src_len)

        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.layernorm1(src+self.dropout(_src))
        _src = self.PositionwiseFeedForwardLayer(src)
        src = self.layernorm2(src+self.dropout(_src))

        return src

        

