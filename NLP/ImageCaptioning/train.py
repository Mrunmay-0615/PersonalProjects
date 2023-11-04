import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples

from dataloader import get_loader
from model import CNNtoRNN

from tqdm import tqdm


def train():

    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader, dataset = get_loader(
        root_folder="data/images",
        annotation_file="data/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)

    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    lr_rate = 3e-4
    num_epochs = 100
    
    # for tensorboard
    writer = SummaryWriter("runs/flicker")
    step = 0

    # initialise the model
    model = CNNtoRNN(vocab_size, hidden_size, num_layers, embed_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_chekcpoint.pth.tar"), model, optimizer)
    
    model.train()
    
    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict" : model.state_dict,
                "optimizer" : optimizer.state_dict,
                "step": step,
            }
            save_checkpoint(checkpoint)
        
        loop = tqdm(train_loader, ncols=150, desc=f'Epoch {epoch+1}')
        total_loss = 0
        for idx, (images, captions) in enumerate(loop):
            imgs = images.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training_loss", loss.item(), global_step=step)
            step+=1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix({"loss":total_loss/(idx+1)})


if __name__ == "__main__":
    train()
        
