#import albumentations as A
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

colorMap = {
    0 : [0, 0, 0], #black
    1 : [158, 11, 11], #dark red
    2 : [193, 179, 51], #dark yellow
    3 : [255, 97, 97], #light red
    4 : [246, 255, 111], #light yellow
    5 : [111, 217, 255], #light blue
    6 : [0, 97, 6], #dark green
    7 : [153, 51, 255], #purple
    8 : [20, 24, 137], #dark blue
    9 : [88, 255, 99] #light green
}


def ImageMask(
                path: str, 
                ID: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        - path: path of the dataset
        - ID: id of the image to visualize


    Returns the image and the corresponding mask with this ID

    """
    pathTotImage=path+"/image/"+str(ID)+".jpg"
    pathTotMask=path+"/mask/"+str(ID)+"_lab"+".png"
    image=Image.open(pathTotImage)
    mask=Image.open(pathTotMask)
    
    return image, mask



def displayMaskGray(
                        mask: np.ndarray
                    ) -> None:
    """
    Displays a mask with grayscale map.

    """
    enhancer = ImageEnhance.Contrast(mask)
    factor = 20 #increase contrast
    mask_cont = enhancer.enhance(factor)
    display(mask_cont)
    
    
    
# 
def displayMask(
                    mask: np.ndarray
                ) -> None:
    """
    Displays a mask with colors (colorMap above).

    """
    mask2=mask
    Amask=np.array(mask2)
    Amask2=np.array([colorMap[Amask[x][y]] for x in range(len(Amask)) for y in range(len(Amask[x]))])
    Amask2=np.resize(Amask2, (Amask.shape[0], Amask.shape[1], 3))
    y=Amask2.shape[0]
    x=Amask2.shape[1]
    plt.figure(figsize=(15, 15*(y/x)))
    plt.imshow(Amask2)


def saveMask(
                mask: np.ndarray, 
                path: str
            ) -> None:
    """
    Saves a mask with colors (colorMap above).

    """
    mask2=mask
    Amask=np.array(mask2)
    Amask2=np.array([colorMap[Amask[x][y]] for x in range(len(Amask)) for y in range(len(Amask[x]))])
    Amask2=np.resize(Amask2, (Amask.shape[0], Amask.shape[1], 3))
    y=Amask2.shape[0]
    x=Amask2.shape[1]
    plt.figure(figsize=(15, 15*(y/x)))
    plt.imshow(Amask2)
    plt.savefig(path)
    
