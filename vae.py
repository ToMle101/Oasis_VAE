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