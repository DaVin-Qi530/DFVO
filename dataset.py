import os
import cv2
import random
from utils import DarkChannel, GuidedFilter
from torch.utils.data import Dataset

class DFDataset(Dataset):
    def __init__(self, transform, dataset="LLVIP", if_train = True, if_patch = True):
        super(DFDataset, self).__init__()
        self.dataset  = dataset
        self.datadirs = os.path.join("Dataset", dataset)
        self.ds       = 10
        self.if_train = if_train
        self.if_patch  = if_patch
        if self.dataset == "MSRS" and not self.if_train:
            self.datadirs = os.path.join(self.datadirs, "eval")
            self.visible = os.path.join(self.datadirs, "visible")
            self.lwir    = os.path.join(self.datadirs, "lwir") 
        else:
            self.visible = os.path.join(self.datadirs, "visible")
            self.lwir    = os.path.join(self.datadirs, "infrared")       
        self.visible_imgs = os.listdir(self.visible)
        self.lwir_imgs    = os.listdir(self.lwir)

        self.transform = transform

    def __len__(self):
        if self.visible_imgs == self.lwir_imgs:
            return len(self.visible_imgs)
        else:
            print("error from dataset!")

    def __getitem__(self, index):
        self.visible_dir = os.path.join(self.visible, self.visible_imgs[index])
        self.lwir_dir    = os.path.join(self.lwir, self.lwir_imgs[index])
        self.visible_img = cv2.imread(self.visible_dir)
        self.visible_img = cv2.cvtColor(self.visible_img, cv2.COLOR_BGR2YCrCb)
        self.lwir_img    = cv2.imread(self.lwir_dir)
        self.lwir_img    = cv2.cvtColor(self.lwir_img, cv2.COLOR_BGR2YCrCb)
        if self.if_patch:
            self.patch_size_h = 256
            self.patch_size_w = 320
            v_img, l_img = self.get_patch(self.visible_img, self.lwir_img)
        else:
            v_img = self.visible_img
            l_img = self.lwir_img

        if self.if_train:
            # dark channel priori
            y_img = v_img[:, :, 0]
            rgb_img = cv2.cvtColor(v_img, cv2.COLOR_YCrCb2RGB)
            dc_img  = DarkChannel(rgb_img, self.ds)
            dc_img  = GuidedFilter(y_img, dc_img, r=30, eps=0.01)
            dc_img = self.transform(dc_img)
            # histogram equalization priori
            his_img = cv2.equalizeHist(y_img)
            his_img = self.transform(his_img)
        
        v_img = self.transform(v_img)
        l_img = self.transform(l_img)

        if not self.if_train:
            return v_img, l_img, self.visible_imgs[index]
        else:
            return v_img, l_img, dc_img, his_img

    def get_patch(self, visible_img, lwir_img):
            H, W = visible_img.shape[:2]
            x = random.randint(0, W - self.patch_size_w)
            y = random.randint(0, H - self.patch_size_h)

            visible_img = visible_img[y:y + self.patch_size_h, x:x + self.patch_size_w, :]
            lwir_img    = lwir_img[y:y + self.patch_size_h, x:x + self.patch_size_w, :]

            return visible_img, lwir_img
        