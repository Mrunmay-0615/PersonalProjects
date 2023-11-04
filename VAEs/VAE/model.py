import torch
import torch.nn as nn
import torch.nn.functional as F

# Input_img -> Hiddenstate -> mean, std -> reparameterisation -> Decoder -> output
class VAE(nn.Module):

    def __init__(self, in_dim, h_dim=200, z_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.img2_hid = nn.Linear(in_dim, h_dim)
        self.hid2_mu = nn.Linear(h_dim, z_dim)
        self.hid2_sigma = nn.Linear(h_dim, z_dim)
        # Decoder
        self.z2_hid = nn.Linear(z_dim, h_dim)
        self.hid2_img = nn.Linear(h_dim, in_dim)

    def encode(self, x):
        # q_phi(z/x)
        x = F.relu(self.img2_hid(x))
        mu = self.hid2_mu(x)
        sigma = self.hid2_sigma(x)
        return mu, sigma

    def decode(self, z):
        # p_theta(x/z)
        h = F.relu(self.z2_hid(z))
        x = torch.sigmoid(self.hid2_img(h))
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = mu + sigma*torch.randn_like(sigma)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma




if __name__ == "__main__":

    x = torch.randn(4, 784)
    vae = VAE(in_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x)