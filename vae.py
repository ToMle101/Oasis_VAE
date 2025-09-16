
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import umap

"""
Dataset calss to load OASIS brain MRI images from PNG files.
"""
class OASISDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        # Get all PNG files from the directory
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load PNG image as grayscale (brain MRI)
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        return image


"""
VAE model components: Encoder, LatentZ, Decoder, and VAE class.
"""
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class LatentZ(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, h):
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, input_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = torch.sigmoid(self.fc2(x))  # output in [0,1]
        return x

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=8):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.latent = LatentZ(hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.latent(h)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


"""
Loss function for VAE
"""
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

"""
Visualization: UMAP
"""
def plot_latent_umap(model, device, dataloader, img_size, fname='latent_umap.png', max_points=2000):
    model.eval()
    mus = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x = batch.view(batch.size(0), -1)
            _, mu, _, _ = model(x)
            mus.append(mu.cpu().numpy())
            if sum(len(m) for m in mus) >= max_points:
                break
    mus = np.concatenate(mus, axis=0)
    mus = mus[:max_points]

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb2d = reducer.fit_transform(mus)

    plt.figure(figsize=(6,6))
    plt.scatter(emb2d[:,0], emb2d[:,1], s=5, alpha=0.7)
    plt.title('Latent space UMAP projection')
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

"""
Training loop
"""
def train_vae(
    data_dir_train, data_dir_test,
    img_size=64, batch_size=128, test_batch=200,
    hidden_size=512, latent_size=8,
    lr=1e-3, epochs=30, beta=1.0,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    trainset = OASISDataset(data_dir_train, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testset = OASISDataset(data_dir_test, transform=transform)
    test_loader = DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=2)

    input_size = img_size * img_size
    model = VAE(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            x = batch.view(batch.size(0), -1)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss/(pbar.n+1)})

        print(f"Epoch {epoch} avg loss per image: {train_loss/len(trainset):.4f}")

    # Final latent visualization
    plot_latent_umap(model, device, test_loader, img_size, fname='latent_umap.png')
    print("Training finished. UMAP saved as latent_umap.png")

    return model

"""
Main
"""
if __name__ == '__main__':
    base_path = '/home/groups/comp3710/OASIS/'
    train_dir = os.path.join(base_path, 'keras_png_slices_train')
    test_dir = os.path.join(base_path, 'keras_png_slices_test')

    model = train_vae(
        data_dir_train=train_dir,
        data_dir_test=test_dir,
        img_size=64,
        batch_size=128,
        test_batch=200,
        hidden_size=1024,
        latent_size=8,    
        lr=1e-3,
        epochs=30,
        beta=1.0
    )