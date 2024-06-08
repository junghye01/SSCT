import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from model_utils import AttentionModule, DeformableConv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import glob
from utils import read_nii_image,compute_sobel_batch_gradients
import wandb
import argparse
from model import S3Net
from torchvision import transforms


class NiiDataset(Dataset):
    def __init__(self,file_list,img_size,transform=None):
        self.file_list=file_list
        self.img_size=img_size
        self.transform=transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        img_file=self.file_list[idx]
        image=read_nii_image(img_file,self.img_size)
        if self.transform:
            image=self.transform(image)

        return image
    

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


def train(model,dataloader,device,epochs,lr,nChannel,loss_ce_coef,loss_at_coef,loss_s_coef):
    model.train()
    label_colours = np.random.randint(255, size=(100, 3))

    loss_ce = torch.nn.CrossEntropyLoss()
    loss_s = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)

            zero_img = torch.zeros(images.shape[2], images.shape[3]).to(device)

            optimizer.zero_grad()
            output, out_auxi = model(images)

            ### Output
            output, out_auxi = output, out_auxi
        

            msk_auxi = torchvision.transforms.functional.affine(output, angle=90, translate=(0, 0), scale=1.0, shear=0.0)
            output = output.permute(0, 2, 3, 1).contiguous().view(-1, nChannel)
            
            _, target = torch.max(output, 1)
            img_target_rgb = np.array([label_colours[c % nChannel] for c in target.cpu().numpy()])

            img_target_rgb = img_target_rgb.reshape(images.shape[0], images.shape[2], images.shape[3], 3).astype(np.uint8)

            ### Cross-entropy loss function
            loss_ce_value = loss_ce_coef * loss_ce(output, target)

            ### Affine transform loss function
            msk_auxi = msk_auxi.permute(0, 2, 3, 1).contiguous().view(-1, nChannel)
            _, target_auxi = torch.max(msk_auxi, 1)
            gt_auxi = out_auxi.permute(0, 2, 3, 1).contiguous().view(-1, nChannel)
            loss_at_value = loss_at_coef * loss_ce(gt_auxi, target_auxi)

            ### Spatial loss function
            sub_y_x, sub_xy_x, sub_xy_y = compute_sobel_batch_gradients(img_target_rgb)
            loss_y_x = loss_s(sub_y_x, zero_img)
            loss_xy_x = loss_s(sub_xy_x, zero_img)
            loss_xy_y = loss_s(sub_xy_y, zero_img)
            loss_s_value = loss_s_coef * (loss_y_x + loss_xy_x + loss_xy_y)

            ### Optimization
            loss = loss_ce_value + loss_at_value + loss_s_value
            loss.backward()
            optimizer.step()

            nLabels = len(np.unique(target.cpu().numpy()))
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)} | Label num: {nLabels} | Loss: {round(loss.item(), 4)} | CE: {round(loss_ce_value.item(), 4)} | AT: {round(loss_at_value.item(), 4)} | Spatial: {round(loss_s_value.item(), 4)}')
    
            wandb.log({
                "epoch": epoch+1,
                "batch_idx": batch_idx+1,
                "nLabels":nLabels,
                "loss":loss.item(),
                "loss_ce":loss_ce_value.item(),
                "loss_at":loss_at_value.item(),
                "loss_spatial":loss_s_value.item()
            })

    # save model
    torch.save(model.state_dict(), f'S3Net_ACDC_{epochs}.pth')
    print(f"Training complete. Model saved as S3Net_ACDC_{epochs}.pth.")


if __name__=='__main__':
    wandb.init(project='S3Net-segmentation')

    parser=argparse.ArgumentParser()
    parser.add_argument('--img_path',type=str,default='/share0/dbs2024_1/ACDC/input_images/train/image/')
    parser.add_argument('--img_size',type=int,default=256)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.05)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--loss_ce_coef',type=float,default=1.0)
    parser.add_argument('--loss_at_coef',type=float,default=0.1)
    parser.add_argument('--loss_s_coef',type=float,default=1.0)
    parser.add_argument('--nChannel',type=int,default=64)
    parser.add_argument('--color',type=bool,default=False)
    parser.add_argument('--nConv',type=int,default=2)

    args=parser.parse_args()

    img_data=sorted(glob.glob(args.img_path + '*'))

    use_cuda=torch.cuda.is_available()
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")

    
    # transform 추가
    dataset = NiiDataset(img_data, args.img_size,transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # model
    model = S3Net(in_channels=3, nChannel=args.nChannel, nConv=args.nConv,
              BHW=(args.batch_size, args.img_size, args.img_size), color=args.color).to(device)

    train(model,dataloader,device,args.epochs,args.lr,args.nChannel,args.loss_ce_coef,args.loss_at_coef,args.loss_s_coef)
