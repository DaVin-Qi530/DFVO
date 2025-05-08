import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def gradient(input_tensor, direction):
    weights = torch.tensor([[0., 0.],
                            [-1., 1.]]
                           ).cuda()
    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
    weights_y = torch.transpose(weights_x, 2, 3)

    if direction == "x":
        weights = weights_x
    elif direction == "y":
        weights = weights_y
    grad_out = torch.abs(F.conv2d(input_tensor, weights, stride=1, padding=1))
    return grad_out

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
    
class RetinexLoss(nn.Module):
    def __init__(self):
        super(RetinexLoss, self).__init__()
    
    def forward(self, r, l, vi):
        return torch.mean(torch.abs((r * l) - vi))

class MGLoss(nn.Module):
    def __init__(self):
        super(MGLoss, self).__init__()

    def forward(self, il):
        il_gx = gradient(il, "x")
        il_gy = gradient(il, "y")
        x = il_gx * torch.exp(-10 * il_gx)
        y = il_gy * torch.exp(-10 * il_gy)
        return torch.mean(x + y)
    
class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, il, vi):
        il_gx = gradient(il, "x")
        il_gy = gradient(il, "y")
        vi_gx = gradient(vi, "x")
        vi_gy = gradient(vi, "y")
        x_loss = torch.abs(torch.div(il_gx, torch.clamp(vi_gx, 0.01)))
        y_loss = torch.abs(torch.div(il_gy, torch.clamp(vi_gy, 0.01)))
        return torch.mean(x_loss + y_loss)
    
class HisLoss(nn.Module):
    def __init__(self):
        super(HisLoss, self).__init__()
        resnet_model = resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(resnet_model.children())[:8])
        self.feature = self.feature.cuda().eval()
        self.l2loss  = nn.MSELoss()

    def forward(self, his_rgb, en_rgb):
        with torch.no_grad():
            his_fea = self.feature(his_rgb)
            en_fea  = self.feature(en_rgb)      
        return self.l2loss(en_fea, his_fea)#Input Target
    
class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        self.Sobelxy = Sobelxy()
        self.l2loss  = nn.MSELoss()
    
    def forward(self, vi_o, dc_o, ir_o, fu):
        with torch.no_grad():
            # replace hight-light
            mean_value  = torch.mean(dc_o)
            vi_ = torch.where(dc_o > 7 * mean_value, ir_o, vi_o).detach()
        # loss function
        vi_grad = self.Sobelxy(vi_)
        ir_grad = self.Sobelxy(ir_o)
        fu_grad = self.Sobelxy(fu)
        x_grad  = torch.max(vi_grad, ir_grad)
        loss =  self.l2loss(fu_grad, x_grad) #Input, Target
        return loss
    
class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.l2loss = nn.MSELoss()
        self.Sobelxy = Sobelxy()

    def forward(self, vi_o, dc_o, ir_o, fu):
        with torch.no_grad():
            vi_g  = self.Sobelxy(vi_o)
            ir_g  = self.Sobelxy(ir_o)
            w_vi  = torch.unsqueeze(torch.mean(vi_g), dim=-1)
            w_ir  = torch.unsqueeze(torch.mean(ir_g), dim=-1)
            # replace hight-light
            mean_value  = torch.mean(dc_o)
            vi_ = torch.where(dc_o > 7 * mean_value, ir_o, vi_o).detach()
        weight_list = torch.cat((w_vi, w_ir), dim=-1)
        weight_list = F.softmax(weight_list, dim=-1)
        # loss function
        loss1 = self.l2loss(fu, vi_)
        loss2 = self.l2loss(fu, ir_o)
        loss  = torch.mean(weight_list[0].item() * loss1 + weight_list[1].item() * loss2)
        return loss

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity()

    def forward(self, vi, fu):
        loss = 1 - self.cos(fu, vi)
        return torch.mean(loss)

class DFLOLoss(nn.Module):
    def __init__(self):
        super(DFLOLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.mgloss = MGLoss()
        self.reloss = RetinexLoss()
        self.smoothloss = SmoothLoss()
        self.hisloss   = HisLoss()
        self.sobelloss = SobelLoss()
        self.contloss  = FusionLoss()
        self.cosloss   = CosineLoss()

    def forward(self, il, en, ir, fu, vi_o, dc_o, ir_o, his_rgb, en_rgb, fu_rgb, num):
        ir_ = ir.detach() # no backward
        en_ = en.detach() # no backward
        en_rgb_ = en_rgb.detach() # no backward
        loss = 10 * self.l1loss(ir, ir_o) + 500 * self.reloss(en, il, vi_o) + 1.35 * self.mgloss(il) \
             + 1.55 * self.smoothloss(il, vi_o) + 2.5 * self.hisloss(his_rgb, en_rgb) \
             + 0.1 * num * (0.65 * self.sobelloss(en_, dc_o, ir_, fu) + 1.75 * self.contloss(en_, dc_o, ir_, fu) \
             + 0.35 * self.cosloss(en_rgb_, fu_rgb))
        return loss