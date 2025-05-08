import torch
import cv2
import numpy as np
from PIL import Image

def YCrCb2RGB(input_im, device):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    if device == torch.device("cuda"):
        mat = torch.tensor(
            [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
        ).cuda()
        bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
        temp = (im_flat + bias).mm(mat).cuda()
    else:
        mat = torch.tensor(
            [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
        )
        bias = torch.tensor([0.0 / 255, -0.5, -0.5])
        temp = (im_flat + bias).mm(mat)
    out = ( temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,).transpose(1, 3).transpose(2, 3))
    return out

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def GuidedFilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
 
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I
 
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I
 
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
 
    q = mean_a*im + mean_b
    return q
    
def SaveImage(img, save_path):
    with torch.no_grad():
        img = torch.clamp(img, 0, 1) # 0~1
        img = img.cpu().numpy()
        img = img.transpose((0, 2, 3, 1))
        img = (img - np.min(img)) / (
            np.max(img) - np.min(img)
        )
        img = np.uint8(255.0 * img)
        img = img.squeeze()
        img = Image.fromarray(img)
        img.save(save_path)