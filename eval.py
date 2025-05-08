import os
import time
import torch
from torchvision import transforms
from modules import SimplifiedModel
from torch.utils.data import DataLoader
from dataset import DFDataset
from utils import YCrCb2RGB, SaveImage

def eval():
    EPS = 1e-8
    save_path   = "eval/"
    FModel_name = "models/DFModel.pth"
    data_transfrom = transforms.Compose([transforms.ToTensor()])
    device      = torch.device("cpu")
    dataset     = DFDataset(data_transfrom, dataset = "test_50", if_train=False, if_patch=False)
    dataloader  = DataLoader(dataset, batch_size=1)
    network     = SimplifiedModel().to(device)
    # load network
    checkpoint  = torch.load(FModel_name)
    model_dict  = network.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    print("Load Fusion Model {} Successfully!".format(FModel_name))
    network.eval()
    actual_times = []
    with torch.no_grad():
        for _,(v_img, l_img, imgName) in enumerate(dataloader):
            v_img = v_img.to(device)
            l_img = l_img.to(device)
            v_img_y = v_img[:, 0, :, :].unsqueeze(1)
            l_img_y = l_img[:, 0, :, :].unsqueeze(1)
            start_time = time.time()
            f_img = network(v_img_y, l_img_y)
            img_cr = v_img[:, 1:2, :, :]
            img_cb = v_img[:, 2:,  :, :]
            w_cr = (torch.abs(img_cr) + EPS) / torch.sum(torch.abs(img_cr) + EPS, dim=1, keepdim=True)
            w_cb = (torch.abs(img_cb) + EPS) / torch.sum(torch.abs(img_cb) + EPS, dim=1, keepdim=True)
            fused_img_cr = torch.sum(w_cr * img_cr, dim=1, keepdim=True).clamp(-1, 1)
            fused_img_cb = torch.sum(w_cb * img_cb, dim=1, keepdim=True).clamp(-1, 1)
            fused_img = torch.cat((f_img, fused_img_cr, fused_img_cb), dim=1) 
            fused_img = YCrCb2RGB(fused_img, device)
            actual_time = time.time() - start_time

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            SaveImage(fused_img, save_path + imgName[0])
            print("Finish num {} image!".format(imgName[0]))
            actual_times.append(actual_time)
    print("Finish eval!!!!")
    print("Using Avg Time is [%lf]" % (sum(actual_times) / len(actual_times)))


if __name__ == "__main__":
    eval()