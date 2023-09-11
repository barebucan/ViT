import os
from PIL import Image
from torch.utils.data import Dataset

import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from config import W,H, P, train_split, device

Trans = T.Compose([
                    T.Resize((W,H)),
                    T.RandomHorizontalFlip(),
                   T.ToTensor(),
                   T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir, json_file,split):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        with open(json_file, 'r') as f:
            class_mapping = json.load(f)
            class_labels = list(class_mapping.keys())  # Get the original class labels
            class_labels.sort()  # Sort the class labels
            class_mapping = {class_labels[i]: i for i in range(len(class_labels))}  # Create a new mapping

        # Iterate over the directories and collect image paths and labels
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if folder_name.startswith(split):
                class_folders = os.listdir(folder_path)
                for class_folder in class_folders:
                    class_dir = os.path.join(folder_path, class_folder)
                    image_files = os.listdir(class_dir)
                    self.image_paths.extend([os.path.join(class_dir, img_file) for img_file in image_files])
                    self.labels.extend([int(class_mapping[class_folder])] * len(image_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = torch.tensor(self.labels[index])
        image = Image.open(image_path).convert('RGB')

        patches = self._patching_(image)

        return patches.to(device),label.to(device)
    
    def image_to_patches_single(self, image, patch_size):
        num_channels, image_height, image_width = image.shape
        unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        patches = unfold(image.unsqueeze(0)).transpose(1, 2).reshape(-1, num_channels, patch_size, patch_size)
        
        # Add an extra patch at the start initialized with all ones
        #extra_patch = torch.ones(1, num_channels, patch_size, patch_size, device=image.device)
        #patches = torch.cat([extra_patch, patches], dim=0)
        
        # Flatten patches
        patches = patches.view(-1, patch_size * patch_size * num_channels)
        
        return patches

    def _patching_(self, image):
        
        img_tensor = Trans(image)
        return self.image_to_patches_single(img_tensor, P)

        
        
