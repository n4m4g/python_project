import torch
from torch import nn


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(dim=1)

    onehot = torch.zeros((idx.size(0), n), device=idx.device)
    # scatter_(dim, index, src, reduce=None)
    # self[i][index[i][j]] = src[i][j]  # if dim == 1
    onehot.scatter_(1, idx, 1)

    return onehot


class VAE(nn.Module):
    def __init__(self, en_layer_size, latent_size,
                 de_layer_size, conditional=False, num_labels=0):

        super(VAE, self).__init__()
        if conditional:
            assert num_labels > 0

        self.latent_size = latent_size
        self.encoder = Encoder(en_layer_size, latent_size,
                               conditional, num_labels)
        self.decoder = Decoder(de_layer_size, latent_size,
                               conditional, num_labels)

    def forward(self, x, c=None):

        # x.shape = (batch_size, 1, 28, 28)
        # c.shape = (batch_size)

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        mean, log_var = self.encoder(x, c)
        # mean.shape = (batch_size, latent_size)
        # log_var.shape = (batch_size, latent_size)

        z = self.reparameterize(mean, log_var)
        # z.shape = (batch_size, latent_size)

        recon_x = self.decoder(z, c)
        # recon_x = (batch_size, layer_size[-1])

        return recon_x, mean, log_var, z

    @staticmethod
    def reparameterize(mean, log_var):

        # mean.shape = (batch_size, latent_size)
        # log_var.shape = (batch_size, latent_size)

        std = torch.exp(0.5 * log_var)
        # std.shape = (batch_size, latent_size)
        eps = torch.randn_like(std)
        # eps.shape = (batch_size, latent_size)
        z = mean + eps * std
        # z.shape = (batch_size, latent_size)

        return z

    def inference(self, z, c=None):

        # z.shape = (batch_size, latent_size)
        # c.shape = (batch_size)

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):
    def __init__(self, layer_size, latent_size,
                 conditional, num_labels):
        super(Encoder, self).__init__()

        self.conditional = conditional
        if self.conditional:
            layer_size[0] += num_labels

        self.num_labels = num_labels

        self.MLP = nn.Sequential()

        for idx in range(len(layer_size[:-1])):
            # append FC
            name = f"fc{idx}"
            module = nn.Linear(*layer_size[idx:idx+2])
            self.MLP.add_module(name=name, module=module)

            # append ReLU
            name = f"relu{idx}"
            module = nn.ReLU()
            self.MLP.add_module(name=name, module=module)

        self.fc_mean = nn.Linear(layer_size[-1], latent_size)
        self.fc_log_var = nn.Linear(layer_size[-1], latent_size)

    def forward(self, x, c=None):
        # x.shape = (batch_size, 28*28)
        # c.shape = (batch_size)

        if self.conditional:
            c = idx2onehot(c, self.num_labels)
            # c.shape = (batch_size, num_labels)
            x = torch.cat((x, c), dim=-1)
            # x.shape = (batch_size, 28*28+num_labels)

        x = self.MLP(x)
        # x.shape = (batch_size, layer_size[-1])

        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        # m.shape = (batch_size, latent_size)
        # s.shape = (batch_size, latent_size)

        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, layer_size, latent_size,
                 conditional, num_labels):
        super(Decoder, self).__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            in_size = latent_size + num_labels
        else:
            in_size = latent_size

        layer_size = [in_size] + layer_size

        for idx in range(len(layer_size[:-1])):
            # append FC
            name = f"fc{idx}"
            module = nn.Linear(*layer_size[idx:idx+2])
            self.MLP.add_module(name=name, module=module)

            if idx+2 < len(layer_size):
                name = f"relu{idx}"
                module = nn.ReLU()
                self.MLP.add_module(name=name, module=module)
            else:
                name = f"sigmoid{idx}"
                module = nn.Sigmoid()
                self.MLP.add_module(name=name, module=module)

    def forward(self, z, c):
        # z.shape = (batch_size, latent_size)
        # c.shape = (batch_size)

        if self.conditional:
            c = idx2onehot(c, self.num_labels)
            # c.shape = (batch_size, num_labels)
            z = torch.cat((z, c), dim=-1)
            # z.shape = (batch_size, latent_size+num_labels)

        x = self.MLP(z)
        # x.shape = (batch_size, layer_size[-1])

        return x


if __name__ == "__main__":
    vae = VAE([784, 256], 10, [256, 784], conditional=True, num_labels=10)
    print(vae)
