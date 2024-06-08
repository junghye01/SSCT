import numpy as np
import cv2
import torch
import nibabel as nib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_image(img_file, img_size):
    im = cv2.imread(img_file)
    im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    
    return data


def read_nii_image(img_file, img_size, slice_index=None):
    """
    Read a NIfTI image file, resize it, and convert to PyTorch tensor.

    :param img_file: Path to the NIfTI image file.
    :param img_size: Desired size of the output image (img_size x img_size).
    :param slice_index: Index of the slice to extract from the 3D image. If None, use the middle slice.
    :return: Torch tensor of the image.
    """
   
    img = nib.load(img_file)
    data = img.get_fdata()

    # default : middle slice
    if slice_index is None:
        slice_index = data.shape[2] // 2
    slice_data = data[:, :, slice_index]

    # resize image
    slice_data_resized = cv2.resize(slice_data, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # 채널 차원 추가 및 텐서로 변환 (3 채널로 복제하여 RGB 형식으로 만듦)
    slice_data_resized = np.stack([slice_data_resized] * 3, axis=0)  # [3, H, W] 형식
    slice_data_resized = slice_data_resized.astype('float32') / 255.0
    #slice_data_resized = slice_data_resized[np.newaxis, :, :, :].astype('float32') / 255.0  # [1, 3, H, W] 형식
    slice_data_resized = torch.from_numpy(slice_data_resized)
    
    return slice_data_resized

def compute_sobel_gradients(img_rgb):
    """
    Compute Sobel gradients on the given RGB image.

    :param img_rgb: Input RGB image.
    :type img_rgb: numpy.ndarray
    :return: Gradients on X, Y, and both X and Y axes.
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor
    """
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)/255.
    
    sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    
    sub_y_x = torch.tensor(sobely - sobelx).to(device)
    sub_xy_x = torch.tensor(sobelxy - sobelx).to(device)
    sub_xy_y = torch.tensor(sobelxy - sobely).to(device)
    
    return sub_y_x, sub_xy_x, sub_xy_y


def compute_sobel_batch_gradients(img_batch_rgb):
    """
    Compute Sobel gradients on the given batch of RGB images.

    :param img_batch_rgb: Batch of input RGB images. Shape: (batch_size, height, width, channels).
    :type img_batch_rgb: numpy.ndarray
    :return: Gradients on X, Y, and both X and Y axes for each image in the batch.
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor
    """
    batch_size = img_batch_rgb.shape[0]
    
    sub_y_x_list, sub_xy_x_list, sub_xy_y_list = [], [], []
    
    for i in range(batch_size):
        img_rgb = img_batch_rgb[i]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) / 255.0

        sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)

        sub_y_x_list.append(sobely - sobelx)
        sub_xy_x_list.append(sobelxy - sobelx)
        sub_xy_y_list.append(sobelxy - sobely)
    
    sub_y_x = torch.tensor(np.array(sub_y_x_list)).to(device)
    sub_xy_x = torch.tensor(np.array(sub_xy_x_list)).to(device)
    sub_xy_y = torch.tensor(np.array(sub_xy_y_list)).to(device)
    
    return sub_y_x, sub_xy_x, sub_xy_y

def create_mask(pred, GT):
    
    kernel = np.ones((5, 5), np.uint8) 
    dilated_GT = cv2.dilate(GT, kernel, iterations = 2)

    mult = pred * GT        
    unique, count = np.unique(mult[mult !=0], return_counts=True)
    cls= unique[np.argmax(count)]
    
    lesion = np.where(pred==cls, 1, 0) * dilated_GT
    
    return lesion


def dice_metric(A, B):
    intersect = np.sum(A * B)
    fsum = np.sum(A)
    ssum = np.sum(B)
    dice = (2 * intersect ) / (fsum + ssum)
    
    return dice    


def hm_metric(A, B):
    intersection = A * B
    union = np.logical_or(A, B)
    hm_score = (np.sum(union) - np.sum(intersection)) / np.sum(union)
    
    return hm_score


def xor_metric(A, GT):
    intersection = A * GT
    union = np.logical_or(A, GT)
    xor_score = (np.sum(union) - np.sum(intersection)) / np.sum(GT)
    
    return xor_score