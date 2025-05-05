# The original code is licensed under MIT License, which is can be found at licenses/LICENSE_UVIT.txt.

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = sorted([os.path.join(root_dir, fname) for fname in os.listdir(root_dir)])
        self.transform = transform or get_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_dataloader(config):
    train_dataset = MedicalImageDataset(root_dir=config['train_data'], transform=None)
    val_dataset = MedicalImageDataset(root_dir=config['val_data'], transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    return train_loader, val_loader

# ----------------------------------------------------------------------------
