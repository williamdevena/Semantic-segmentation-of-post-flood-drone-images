import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2

class FloodNetDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir=root_dir
        self.transform=transform
        self.imagesMasks=self.datasetImagesMasks()
        print("Length of dataset: " + str(self.__len__()))
        print("Shape of images: " + str(self.__getitem__(0)[0].shape))
        print("Shape of masks: " + str(self.__getitem__(0)[1].shape))

        
    def __len__(self) -> int:
        return len(self.imagesMasks)


    def __getitem__(self, 
                    index: int
                    ) -> tuple[np.ndarray, np.ndarray]:

        img = self.imagesMasks[index][0]
        mask = self.imagesMasks[index][1]

        return img, mask

    
    def datasetImagesMasks(self) -> List:
    """
    Collects all the file names of the images and of the masks of the dataset into an array composed by tuple of (image, mask).

    """
        print("COLLECTING IMAGES AND MASKS...")
        images_masks=[]
        x=0
        for path, subdirs, files in os.walk(self.root_dir):
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'image':   #Find all 'images' directories
                images = os.listdir(path)  #List of all image names in this subdirectory
                for i, image_name in enumerate(images):  
                    if image_name.endswith(".jpg"):   #Only read jpg images...  
                        x+=1
                        print("\r", x, sep="", end="", flush=True)  
                        img = cv2.imread(self.root_dir+"/"+"image/"+image_name)
                        mask_name = image_name.split('.')[0]+"_lab.png"
                        mask = cv2.imread(self.root_dir+"/"+"mask/"+mask_name, 0) 
                        augmentations = self.transform(image=img, mask=mask)
                        img = augmentations['image']
                        mask = augmentations['mask']        
                        img = img/255
                        images_masks.append((img, mask, image_name))

        return sorted(images_masks, key=lambda x: x[2])
    
      
  
    