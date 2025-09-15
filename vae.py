import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.optim as optim
from PIL import Image
import os
import glob

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

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()  
])

# Use the existing train/validation/test splits
base_path = '/home/groups/comp3710/OASIS/'

trainset = OASISDataset(
    data_dir=base_path + 'keras_png_slices_train/',
    transform=transform
)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = OASISDataset(
    data_dir=base_path + 'keras_png_slices_test/',
    transform=transform
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        p_x = F.relu(self.fc1(x))
        p_x = F.relu(self.fc2(p_x))

        return p_x


class LatentZ(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, p_x):
        mu = self.mu(p_x)
        logvar = self.logvar(p_x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return std * eps + mu, logvar, mu


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, input_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, z_x):
        q_x = F.relu(self.fc1(z_x))
        q_x = torch.sigmoid(self.fc2(q_x))

        return q_x


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = Encoder(input_size, hidden_size)
        self.latent_z = LatentZ(hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        p_x = self.encoder(x)
        z, logvar, mu = self.latent_z(p_x)
        q_z = self.decoder(z)

        return q_z, logvar, mu, z