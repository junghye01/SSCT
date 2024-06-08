import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import argparse
from model import S3Net
from model_utils import AttentionModule, DeformableConv
from utils import read_nii_image, compute_sobel_gradients
from torchvision import transforms


class NiiDataset(Dataset):
    def __init__(self, img_files, gt_files, img_size,transform=None):
        self.img_files = img_files
        self.gt_files = gt_files
        self.img_size = img_size
        self.transform=transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        gt_file = self.gt_files[idx]
        image = read_nii_image(img_file, self.img_size)
        gt_image = read_nii_image(gt_file, self.img_size)

        if self.transform:
            image=self.transform(image)
            gt_image=self.transform(gt_image)

        return image, gt_image



def dice_score2(pred, target, smooth=1.0):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def dice_score(pred, target, smooth=1.0):
    """
    Compute the Dice Score.

    :param pred: Predicted binary mask.
    :param target: Ground truth binary mask.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Dice score.
    """
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate(model, dataloader, device, nChannel):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for images, gt_masks in dataloader:
            images = images.to(device)
            gt_masks = gt_masks.to(device)

            output, _ = model(images)
            # sigmoid
            output=torch.sigmoid(output)
            output = output.permute(0, 2, 3, 1).contiguous().view(-1, nChannel)
            _, predicted = torch.max(output, 1)
            predicted = predicted.view(images.size(0), images.size(2), images.size(3))

            for i in range(images.size(0)):
                
                pred_mask = predicted[i].cpu().numpy()
                gt_mask = gt_masks[i].cpu().numpy().squeeze()  # Remove channel dimension if it exists
                gt_mask = cv2.resize(gt_mask, (images.size(3), images.size(2)), interpolation=cv2.INTER_NEAREST)
                gt_mask = torch.tensor((gt_mask > 0).astype(np.float32))
                pred_mask = torch.tensor((pred_mask > 0).astype(np.float32))
                #gt_mask = (gt_mask > 0).astype(np.float32)  # Convert to binary mask
                #pred_mask = (pred_mask > 0).astype(np.float32)  # Convert to binary mask

                dice = dice_score2(pred_mask, gt_mask)
                dice_scores.append(dice)

    avg_dice_score = np.mean(dice_scores)
    return avg_dice_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/share0/dbs2024_1/ACDC/input_images/test/image/')
    parser.add_argument('--gt_path', type=str, default='/share0/dbs2024_1/ACDC/input_images/test/GT/')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--nChannel', type=int, default=64)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--nConv', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='S3Net_ACDC_10.pth')

    args = parser.parse_args()

    img_files = sorted(glob.glob(os.path.join(args.img_path, '*.nii.gz')))
    gt_files = sorted(glob.glob(os.path.join(args.gt_path, '*.nii.gz')))

    assert len(img_files) == len(gt_files), "The number of images and ground truth files should be the same."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = NiiDataset(img_files, gt_files, args.img_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = S3Net(in_channels=3, nChannel=args.nChannel, nConv=args.nConv,
                  BHW=(args.batch_size, args.img_size, args.img_size), color=args.color).to(device)
    model.load_state_dict(torch.load(args.model_path))

    avg_dice = evaluate(model, test_dataloader, device, args.nChannel)
    print(f'Average Dice Score: {avg_dice}')

