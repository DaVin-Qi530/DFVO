import os
import torch
import argparse
from torchvision import transforms
from dataset import DFDataset
from torch.utils.data import DataLoader
from modules import DFLONet
from loss import DFLOLoss
from rich import progress
from torch.optim import Adam
from utils import YCrCb2RGB, SaveImage

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transfrom = transforms.Compose([transforms.ToTensor()])
    train_dataset = DFDataset(data_transfrom, dataset = args.train_dataset, if_train=True, if_patch=args.if_patch)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=False)

    DFModelName = os.path.join(args.model_path, args.model)
    network = DFLONet().to(device)
    df_loss = DFLOLoss().to(device)
    optim   = Adam(network.parameters(), lr=1e-4, betas=[0.9, 0.8])
    network.train()
    for epoch in range(args.epochs):
        loop = progress.track(enumerate(dataloader), \
                              description="Training[{}]..".format(epoch+1), total=len(dataloader))
        for _, (v_img, l_img, dc_img, his_img) in loop:
            v_img = v_img.to(device)
            l_img = l_img.to(device)
            dc_img = dc_img.to(device)
            his_img = his_img.to(device)
            v_img_y = v_img[:, 0, :, :].unsqueeze(1)
            l_img_y = l_img[:, 0, :, :].unsqueeze(1)
            [il, en, ir, fu] = network(v_img_y, l_img_y)#vi_o, ir_o
            #get rgb
            en_rgb = YCrCb2RGB(torch.cat((en, v_img[:, 1:, :, :]), dim=1), device)
            vi_rgb = YCrCb2RGB(torch.cat((v_img_y, v_img[:, 1:, :, :]), dim=1), device)
            his_rgb = YCrCb2RGB(torch.cat((his_img, v_img[:, 1:, :, :]), dim=1), device)
            fu_rgb = YCrCb2RGB(torch.cat((fu, v_img[:, 1:, :, :]), dim=1), device)
            #optim
            optim.zero_grad()
            loss = df_loss(il, en, ir, fu, v_img_y, dc_img, l_img_y, \
                           his_rgb, en_rgb, fu_rgb, epoch)
            loss.backward()
            optim.step() 
        
        if (epoch + 1) % 5 == 0: 
            state = {
                'model': network.state_dict()
            }
            torch.save(state, DFModelName)
            #save image
            SaveImage(en_rgb, "trained/Enhance_{}.png".format(epoch + 1))
            SaveImage(ir, "trained/IR_{}.png".format(epoch + 1))
            SaveImage(fu_rgb, "trained/Fusion_{}.png".format(epoch + 1))
            SaveImage(vi_rgb, "trained/OriRGB_{}.png".format(epoch + 1))
            SaveImage(l_img_y, "trained/OriIR_{}.png".format(epoch + 1))

def option():
    parser = argparse.ArgumentParser(description='DFVO')

    parser.add_argument('--epochs', type=int, default=100, 
                        help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='training batch size')    
    parser.add_argument('--train_dataset', type=str, default='LLVIP',
                        help='training dataset directory')    
    parser.add_argument('--model_path', type=str, default='./models',
                        help='trained model directory')    
    parser.add_argument('--model', type=str, default='DFModel.pth',
                        help='model name')
    parser.add_argument('--if_patch', type=bool, default=True, 
                        help='patch the origin image')
    return parser.parse_args()

if __name__ == "__main__":
    opt = option()
    train(opt)