import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class OdometerDataset(Dataset):
    def __init__(self, file, image_dir, transform=None):
        """
        Odometer dataset for classification
        
        :param file: the path of txt file with image names and labels
        :param image_dir: the path of directory with all images
        :param transform: Optional transform to be applied
        """
        self.image_dir = image_dir
        self.transform = transform
        
        with open(file, 'r') as f:
            self.data = f.readlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        line = self.data[idx].strip()
        img_name, label = line.split(',')
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.tensor(int(label), dtype=torch.long)
        
        return image, label