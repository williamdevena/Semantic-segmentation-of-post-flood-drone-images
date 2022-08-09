import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2

class FloodNetDatasetDiskOld(Dataset):
    
    def __init__(self, root_dir, normalize, transform=None):
        super().__init__()
        self.root_dir=root_dir
        self.transform=transform
        self.normalize=normalize
        self.images=self.datasetImages()
        self.masks=self.datasetMasks()
        print(self.__len__())

        
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image_name = self.images[index]
        mask_name = image_name.split('.')[0]+"_lab.png"   
        img = Image.open(self.root_dir+"/"+"image/"+image_name)
        mask = Image.open(self.root_dir+"/"+"mask/"+mask_name)
        img=np.array(img)
        mask=np.array(mask)
        augmentations = self.transform(image=img, mask=mask)
        img = augmentations['image']
        mask = augmentations['mask']        

        if self.normalize:
            img = img/255

        return img, mask

    
    def datasetImages(self):
        print('PATCHIFYING IMAGES...')
        image_dataset = []
        for path, subdirs, files in os.walk(self.root_dir):
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'image':   #Find all 'images' directories
                images = os.listdir(path)  #List of all image names in this subdirectory
                for i, image_name in enumerate(images):  
                    if image_name.endswith(".jpg"):   #Only read jpg images...
                        image_dataset.append(image_name)

        return sorted(image_dataset, key=lambda x: int(x.split(".")[0]))
    
      
    def datasetMasks(self):
        print('PATCHIFYING MASKS...')
        mask_dataset = []  
        for path, subdirs, files in os.walk(self.root_dir): 
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'mask':   #Find all 'images' directories
                masks = os.listdir(path)  #List of all image names in this subdirectory
                for i, mask_name in enumerate(masks):  
                    if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)         
                        mask_dataset.append(mask_name)

        return sorted(mask_dataset, key=lambda x: int(x.split("_")[0]))
  