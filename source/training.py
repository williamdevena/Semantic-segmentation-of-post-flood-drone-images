from preprocess import *
from dataset import FloodNetDataset
from dataset_disk import FloodNetDatasetDisk
from focalloss import FocalLoss
from attention_unet import *
from importlib import reload
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms.transforms import ToTensor
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Union


def displayMaskId(
                    ds: Union[FloodNetDataset, FloodNetDatasetDisk], 
                    id_image: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        - ds: dataset
        - id_image: id of the image to display


    Displays image from the dataset with this ID

    """
    img, mask = ds.__getitem__(id_image)
    displayMask(mask)

    return img, mask


def displayPrediction(
                        model: nn.Module, 
                        device: str, 
                        img: torch.Tensor
                     ) -> np.ndarray:
    """
    Args:
        - model: model to use to make prediction
        - device: 'cpu' or 'cuda' (gpu)
        - img: input for the model


    Displays the prediction of the model

    """
    out = model(torch.unsqueeze(img,0).to(device))['out']
    out = torch.argmax(out, dim=1)
    displayMask(torch.squeeze(out,0).cpu())

    return out



def get_deeplab_trained(
                            device: str, 
                            path_model: str = "", 
                            trained: bool = True
                        ) -> nn.Module:
    """
    Args:
        - device: 'cpu' or 'cuda' (gpu)
        - path_model: path of the pretrained weights to load on model 
        - trained: if True, the weights in path_model are loaded


    Returns DeepLabV3 model. If trained=True loads parameters from .pth file in path_model

    """
    model = models.segmentation.deeplabv3_resnet101(pretrained= False, progress = True, num_classes = 10, pretrained_backbone = False).to(device)
    if trained:
      print(model.load_state_dict(torch.load(path_model, map_location=torch.device(device))))

    model.train()

    return model



def get_loaders(
                    train_path: str, 
                    val_path: str, 
                    batch_size: int, 
                    train_transform: A.Compose, 
                    val_transform: A.Compose, 
                    shuffle: bool, 
                    from_disk: bool = False
                ) -> tuple[
                            Union[FloodNetDataset ,FloodNetDatasetDisk], 
                            Union[FloodNetDataset ,FloodNetDatasetDisk],
                            DataLoader,
                            DataLoader
                            ]:
    """
    Args:
        - train_path: path of the training dataset
        - val_path: path of the validation dataset
        - batch_size: batch size for training
        - train_transform: Albumentations transformation to use for training
        - val_transform: Albumentations transformation to use for validation
        - shuffle: shuffle parameter for the training dataloader
        - from_disk: if True, the dataset returned is of type FloodNetDatasetDisk (FloodNetDataset otherwise)


    Returns dataloaders and datasets for training and validation

    """
    if from_disk:
        train_dataset = FloodNetDatasetDisk(train_path, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

        val_dataset = FloodNetDatasetDisk(val_path, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    else:
        train_dataset = FloodNetDataset(train_path, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

        val_dataset = FloodNetDataset(val_path, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    return train_dataset, val_dataset, train_dataloader, val_dataloader




def save_model(
                model_state_dict, 
                path
            ):
    """
    Saves the parameters of the model in path.

    """
    torch.save(model_state_dict, path)


def save_plot_losses(
                        losses, 
                        path
                    ):
    """
    Saves plot of the losses.

    """
    plt.clf()
    plt.grid()
    plt.plot(losses)
    plt.savefig(path)




def save_average_loss(
                        epoch, 
                        losses, 
                        path
                    ):
    """
    Writes loss of one epoch.

    """
    with open(path, 'a') as f:
        f.write('Epoch ' + str(epoch))
        f.write('Average loss: ' + str(sum(losses)/len(losses)) + "\n")




# def save_num_erros(
#                     errors1, 
#                     errors2, 
#                     path
#                 ):
#     with open(path, 'a') as f:
#         f.write('Num. images with loss>2: ' + str(len(errors1)) + "\n" + 'Num. images with loss>4: ' + str(len(errors2)) + "\n" + "----------------------" + "\n" + "----------------------" + "\n")





def save_log_train(
                    epoch, 
                    losses, 
                    errors1, 
                    errors2, 
                    model_state_dict, 
                    path_losses, 
                    path_plot, 
                    path_model
                ):
    """
    Saves logs of the training (plots of the loss, saves the model, ...)

    """
    save_model(model_state_dict, path_model)   
    save_average_loss(epoch, losses, path_losses)
    save_num_erros(errors1, errors2, path_losses)
    save_plot_losses(losses, path_plot)





def save_log_val(
                    epoch, 
                    losses, 
                    errors1, 
                    errors2, 
                    path_losses, 
                    path_plot
                ):
    """
    Saves logs of the validation.

    """
    save_average_loss(epoch, losses, path_losses)
    save_num_erros(errors1, errors2, path_losses)
    save_plot_losses(losses, path_plot)




def progress(
                percent, 
                width
            ):
    width2 = int(width/10)
    percent2 = int(percent/10)
    left = width2 * percent2 // 100
    right = width2 - left
    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}"
    
    print("\r[", tags, spaces, "] ", percents," / ", width, sep="", end="", flush=True)






def train_one_epoch(
                        dataloader, 
                        device, 
                        model, loss_fn, 
                        optimizer, 
                        start_epoch, 
                        epoch, 
                        path_model, 
                        path_plot, 
                        path_losses
                    ):
    """
    Executes one epoch of training.

    """

    running_loss = 0.
    last_loss = 0.
    losses=[]
    sum_loss=0
    start_time = time.time()
    worst_images_1 = []
    worst_images_2 = []

    model.train()

    for i, data in enumerate(dataloader):   
        progress(i, len(dataloader)) 
        inputs, masks = data
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = loss_fn(outputs, masks.long())

        losses.append(round(loss.item(),8))
        if loss.item()>2:
            worst_images_1.append(i)
        if loss.item()>4:
            worst_images_2.append(i)

        loss.backward()
        optimizer.step()
    
    print("--- Training time: %s seconds ---" % (time.time() - start_time))
    path_plot = path_plot + "_" + str(start_epoch+epoch+1) + '.png'
    path_model = path_model + "_" + str(start_epoch+epoch+1) + '.pth'

    save_log_train(epoch=start_epoch+epoch+1, losses=losses, errors1=worst_images_1, errors2=worst_images_2,
                   model_state_dict=model.state_dict(), path_losses=path_losses,
                   path_plot=path_plot, path_model=path_model)

    return losses





def validation_loss(
                        dataloader, 
                        device, 
                        model, 
                        loss_fn, 
                        start_epoch, 
                        epoch, 
                        path_plot, 
                        path_losses
                    ):
    """
    Calculates validation loss.

    """
    losses = []
    worst_images_1 = []
    worst_images_2 = []
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            progress(i, len(dataloader))  
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)['out']
            loss = loss_fn(outputs, masks.long())

            losses.append(round(loss.item(),8))

            if loss.item()>2:
                worst_images_1.append(i)
            if loss.item()>4:
                worst_images_2.append(i)


    print("--- Validation time: %s seconds ---" % (time.time() - start_time))
    path_plot = path_plot + "_" + str(start_epoch+epoch+1) + '.png'


    save_log_val(epoch=start_epoch+epoch+1, losses=losses, errors1=worst_images_1, errors2=worst_images_2,
                 path_losses=path_losses, path_plot=path_plot)
    

    return losses




def accuracy(
                output, 
                mask
            ):
    """
    Calculates accuracy of a prediction.

    """
    intersection = [output==mask]
    total_pixels = mask.shape[0]*mask.shape[1]
    return np.count_nonzero(intersection)/total_pixels




def calculateAccuracy(
                        ds, 
                        model, 
                        device
                    ):
    """
    Calculates accuracy on entire dataset.

    """
    model.eval()

    acc_tot=0
    print("Calculating Accuracy ")
    for x in range(len(ds)):
        print("\r[", x, "\\", len(ds), "] ", sep="", end="", flush=True)
        img, mask = ds.__getitem__(x)

        out = model(torch.unsqueeze(img,0).to(device))['out']#
        out = torch.argmax(out, dim=1)
        out = out.squeeze(0)

        if device=='cuda':
            out = out.cpu()

        acc=accuracy(output=np.array(out), mask=np.array(mask))
        print(acc)
        acc_tot += acc

    return (acc_tot/len(ds))







def IoU(
            out, 
            mask, 
            label
        ):
    """
    Calculates IoU for a class on a single image.

    """

     x = np.where(out==label, 1, 0)
     y = np.where(mask==label, 1, 0)

     if np.sum(y)==0 and np.sum(x)==0:
         return None

     summ = x+y
     tp = np.count_nonzero((summ==2))
     fn_fp = np.count_nonzero(summ==1)

     return tp/(tp+fn_fp+1e-6)







def mIoU(
            out, 
            mask, 
            num_classes
        ):
    """
    Calculates mIoU on a single image.

    """
    dict_iou={}
    total=0
    for x in range(1, num_classes):
        iou=IoU(out, mask, x)
        if iou==None:
            continue
        dict_iou[x]=iou
        total+=iou

    meanIoU = total/len(dict_iou)

    return dict_iou, meanIoU






def calculatemIoU(
                    ds, 
                    model, 
                    device
                ):
    """
    Calculates mIoU on a entire dataset.

    """
    dict_classes={1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    start_time = time.time()

    model.eval()
    print("Calculating mIoU ")
    for x in range(len(ds)):
        print("\r[", x, "\\", len(ds), "] ", sep="", end="", flush=True)
        img, mask = ds.__getitem__(x)

        out = model(torch.unsqueeze(img,0).to(device))['out']
        out = torch.argmax(out, dim=1)

        if device=='cuda':
            out = out.cpu()

        dict_iou, _ = mIoU(out, mask, 10)

        for pixel_class in dict_iou:
            dict_classes[pixel_class].append(dict_iou[pixel_class])


    total_meanIoU=0
    for sem_class in dict_classes:
        total_meanIoU += np.average(dict_classes[sem_class])
        print("Total IoU of class " + str(sem_class) + ": " + str(np.average(dict_classes[sem_class])))

    print("--- mIoU time: %s seconds ---" % (time.time() - start_time))

    divide=9

    print("Mean IoU: " + str(total_meanIoU/divide))


    return (total_meanIoU/divide), dict_classes











def train_epochs(
                    device, 
                    train_dataloader, 
                    val_dataloader, 
                    test_ds, 
                    model, 
                    loss_fn, 
                    optimizer,
                    start_epoch, 
                    epochs, 
                    path_model, 
                    path_plot_train, 
                    path_plot_val,
                    path_plot_final, 
                    path_losses_train, 
                    path_losses_val
                ):
    """
    Executes the training of a model for several epochs.

    """

    losses_train_epochs = []
    losses_val_epochs = []

    mIoU_mean = [] 
    mIoU_1 = []
    mIoU_2 = []
    mIoU_3 = []
    mIoU_4 = []
    mIoU_5 = []
    mIoU_6 = []
    mIoU_7 = []
    mIoU_8 = []
    mIoU_9 = []


      
    for epoch in range(epochs):
        print("-------------- Epoch %s --------------" % (epoch+start_epoch+1))
        print("-------------- TRAINING --------------")
        losses_train = train_one_epoch(dataloader=train_dataloader, device=device, model=model, loss_fn=loss_fn, 
                        optimizer=optimizer, start_epoch=start_epoch, epoch=epoch, 
                        path_model=path_model, path_plot=path_plot_train, path_losses=path_losses_train)


        print("-------------- VALIDATION --------------")
        losses_val = validation_loss(dataloader=val_dataloader, device=device, model=model, loss_fn=loss_fn, 
                        start_epoch=start_epoch, epoch=epoch, path_plot=path_plot_val,
                        path_losses=path_losses_val)

        losses_train_epochs.append(sum(losses_train)/len(losses_train))
        losses_val_epochs.append(sum(losses_val)/len(losses_val))

        ### PLOT TRAIN LOSS
        plt.clf()
        plt.grid()
        plt.plot(losses_train_epochs)
        plt.savefig(path_plot_final+"/loss_train/"+str(epoch+start_epoch+1)+".png")

        ### PLOT VAL LOSS
        plt.clf()
        plt.grid()
        plt.plot(losses_val_epochs)
        plt.savefig(path_plot_final+"/loss_val/"+str(epoch+start_epoch+1)+".png")

        ### PLOT TRAIN AND VAL LOSS
        plt.clf()
        plt.grid()
        plt.plot(losses_val_epochs)
        plt.plot(losses_train_epochs)
        plt.savefig(path_plot_final+"/loss_train_val/"+str(epoch+start_epoch+1)+".png")

        ### CALCULATE mIoU
        mIoU, dict_classes = calculatemIoU(test_ds, model, device)
        mIoU_mean.append(mIoU)

        class_1 = np.average(dict_classes[1])
        class_2 = np.average(dict_classes[2])
        class_3 = np.average(dict_classes[3])
        class_4 = np.average(dict_classes[4])
        class_5 = np.average(dict_classes[5])
        class_6 = np.average(dict_classes[6])
        class_7 = np.average(dict_classes[7])
        class_8 = np.average(dict_classes[8])
        class_9 = np.average(dict_classes[9])


        mIoU_1.append(class_1)
        mIoU_2.append(class_2)
        mIoU_3.append(class_3)
        mIoU_4.append(class_4)
        mIoU_5.append(class_5)
        mIoU_6.append(class_6)
        mIoU_7.append(class_7)
        mIoU_8.append(class_8)
        mIoU_9.append(class_9)



        ### PLOT TEST mIoU
        plt.clf()
        plt.grid()
        plt.plot(mIoU_mean)
        plt.savefig("../TRAINING_LOGS/mIoU/mean/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_1)
        plt.savefig("../TRAINING_LOGS/mIoU/1/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_2)
        plt.savefig("../TRAINING_LOGS/mIoU/2/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_3)
        plt.savefig("../TRAINING_LOGS/mIoU/3/"+str(epoch+start_epoch+1)+".png")
    
        plt.clf()
        plt.grid()
        plt.plot(mIoU_4)
        plt.savefig("../TRAINING_LOGS/mIoU/4/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_5)
        plt.savefig("../TRAINING_LOGS/mIoU/5/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_6)
        plt.savefig("../TRAINING_LOGS/mIoU/6/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_7)
        plt.savefig("../TRAINING_LOGS/mIoU/7/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_8)
        plt.savefig("../TRAINING_LOGS/mIoU/8/"+str(epoch+start_epoch+1)+".png")

        plt.clf()
        plt.grid()
        plt.plot(mIoU_9)
        plt.savefig("../TRAINING_LOGS/mIoU/9/"+str(epoch+start_epoch+1)+".png")


        ### WRITE mIoU
        with open("../TRAINING_LOGS/mIoU.txt", 'a') as f:
            f.write('Epoch ' + str(epoch+start_epoch+1) + ":"+"\n")  
            f.write('mIoU: '  + str(mIoU) +"\n")
            f.write('Class 1: '  + str(class_1) +"\n")
            f.write('Class 2: '  + str(class_2) +"\n")
            f.write('Class 3: '  + str(class_3) +"\n")
            f.write('Class 4: '  + str(class_4) +"\n")
            f.write('Class 5: '  + str(class_5) +"\n")
            f.write('Class 6: '  + str(class_6) +"\n")
            f.write('Class 7: '  + str(class_7) +"\n")
            f.write('Class 8: '  + str(class_8) +"\n")
            f.write('Class 9: '  + str(class_9) +"\n")
            f.write("----------------------\n")
            f.write("----------------------\n")



        print("------------------------")
        print("------------------------")
        print("------------------------")
        print("------------------------")
























