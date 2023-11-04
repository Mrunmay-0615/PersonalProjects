import torch
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import VAE

# Set the configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR_RATE = 1e-4 # Karpathy Constant

# Dataset loading
train_data = datasets.MNIST(root='dataset/', transform=transforms.ToTensor(), train=True, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = VAE(in_dim=INPUT_DIM, h_dim=H_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
criterion = nn.BCELoss(reduction='sum')

# Start Training
import sys
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)

        # Loss
        reconstruction_loss = criterion(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))
        loss = reconstruction_loss + kl_div

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


def inference(digit, num_examples):
    """
    :param digit: The digit for which the samples will be generated
    :param num_examples: the number of samples to be generated
    :return:
    """
    images = []
    idx = 0
    for x, y in train_data:
        if y==idx:
            images.append(x)
            idx+=1
        if idx==10:
            break


    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        episilon = torch.randn_like(sigma)
        z = mu + sigma*episilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"gen_images/generated_{digit}_example{example}.png")


for idx in range(10):
    inference(idx, 5)