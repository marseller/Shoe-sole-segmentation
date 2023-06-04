
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
import os

class ShoeDataset(Dataset):
    def __init__(self,image_folders_list):
        self.image_paths = []
        self.mask_paths = []
        for folder in image_folders_list: # fodler - \\Sneakers and Athletic Shoes\\adidas Golf
            images = os.listdir(folder)
            masks = os.listdir(os.path.join(folder,'segmentation'))
            for image in images:
                mask_potential = image[:image.rfind('.')]+'.png'
                if mask_potential in masks:
                    self.image_paths.append(os.path.join(folder,image))
                    self.mask_paths.append(os.path.join(folder,'segmentation',mask_potential))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        img_ar = Image.open(img_path).convert('RGB')
        mask_ar = Image.open(mask_path).convert('L')
        
        return img_ar,mask_ar


def hue_augmentation(dataset,scale = 2):
    augmented_data = []
    
    transform_hue = transforms.Compose([
    transforms.ColorJitter(hue=(-0.5, 0.5)),
    transforms.ToTensor(),
    transforms.Resize((128, 128))])

    tensor =transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
    ])
    for image, mask in dataset:
        augmented_data.append((tensor(image),tensor(mask)))
        for _ in range(scale):
            augmented_data.append((transform_hue(image), tensor(mask)))
    return augmented_data