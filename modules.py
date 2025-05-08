import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from thop import profile

class Conv2dBnLeakyRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dBnLeakyRelu, self).__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Bn   = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        self.LeakyRelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        return self.LeakyRelu(x)
    
class Conv2dSigmoid(nn.Module):
    def __init__(self, in_channels):
        super(Conv2dSigmoid, self).__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels = 1, kernel_size=3, stride=1, padding=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Conv(x)
        return self.Sigmoid(x)
    
class UpConv2dBnLeakyRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpConv2dBnLeakyRelu, self).__init__()
        self.Conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Bn   = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        self.LeakyRelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        return self.LeakyRelu(x)

class SobelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SobelConv, self).__init__()
        soble_filter = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
        self.convx = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, stride = 1, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(soble_filter))
        self.convy = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, stride = 1, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(soble_filter.T))
        self.Conv  = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
    
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        x = self.Conv(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self, hidden_dim=64):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=hidden_dim, oup=hidden_dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=hidden_dim, oup=hidden_dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=hidden_dim, oup=hidden_dim, expand_ratio=2)
        self.shffleconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2    

######################modules######################
class CrossSelfAttention(nn.Module):
    def __init__(self, hid_channels):
        super(CrossSelfAttention, self).__init__()
        #query/key and value
        self.ViV_conv = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.ViSK_conv = SobelConv(hid_channels, hid_channels//8, kernel_size=1)
        self.CrossSQ_conv = SobelConv(hid_channels * 2, hid_channels//8, kernel_size=1)
        self.IrSK_conv = SobelConv(hid_channels, hid_channels//8, kernel_size=1)
        self.IrV_conv = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, vi, ir):
        B, C, H, W = vi.size()
        xq = self.CrossSQ_conv(torch.cat((vi, ir), dim=1)).view(B, -1, H*W).permute(0, 2, 1)
        vi_xk = self.ViSK_conv(vi).view(B, -1, H*W)
        vi_xv = self.ViV_conv(vi).view(B, -1, H*W)
        ir_xk = self.IrSK_conv(ir).view(B, -1, H*W)
        ir_xv = self.IrV_conv(ir).view(B, -1, H*W)
        #attention
        vi_attention = self.softmax(torch.bmm(xq, vi_xk))
        ir_attention = self.softmax(torch.bmm(xq, ir_xk))
        #out
        vi_out = torch.bmm(vi_xv, ir_attention.permute(0, 2, 1))
        ir_out = torch.bmm(ir_xv, vi_attention.permute(0, 2, 1))
        vi_out = vi_out.view(B, C, H, W)
        ir_out = ir_out.view(B, C, H, W)
        return vi_out, ir_out

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(hidden_dim=64) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

######################networks######################    
class ILNet(nn.Module):
    def __init__(self):
        super(ILNet, self).__init__()
        #encode
        self.Encode1 = Conv2dBnLeakyRelu(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.Encode2 = Conv2dBnLeakyRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Encode3 = Conv2dBnLeakyRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #decode
        self.Decode1 = UpConv2dBnLeakyRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Decode2 = UpConv2dBnLeakyRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.Decode3 = UpConv2dBnLeakyRelu(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.Sigmoid = Conv2dSigmoid(in_channels=16)

    def forward(self, x):
        x = self.Encode1(x)
        x = self.pool1(x)
        x = self.Encode2(x)
        x = self.Encode3(x)
        x = self.pool3(x)
        x = self.Decode1(x)
        x = self.Decode2(x)
        x = self.Decode3(x)
        il = self.Sigmoid(x)
        return il

class DFLONet(nn.Module):
    def __init__(self):
        super(DFLONet, self).__init__()
        self.IL     = ILNet()
        #vi encode
        self.ViEncode1 = Conv2dBnLeakyRelu(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.ViEncode2 = Conv2dBnLeakyRelu(in_channels=16, out_channels = 32, kernel_size=3, stride=1, padding=1)
        self.ViEncode3 = Conv2dBnLeakyRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ViEncode4 = Conv2dBnLeakyRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #ir encode
        self.IrEncode1 = Conv2dBnLeakyRelu(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.IrEncode2 = Conv2dBnLeakyRelu(in_channels=16, out_channels = 32, kernel_size=3, stride=1, padding=1)
        self.IrEncode3 = Conv2dBnLeakyRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.IrEncode4 = Conv2dBnLeakyRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #crossattaction
        self.ViConv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=4, padding=1)
        self.ViPool  = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.IrConv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=4, padding=1)
        self.IrPool  = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.CSA     = CrossSelfAttention(hid_channels=256)
        self.ViUpConv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=4, padding=1, dilation=2, output_padding=1)
        self.IrUpConv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=4, padding=1, dilation=2, output_padding=1)
        #detailfeature
        self.ViDetail = DetailFeatureExtraction()
        self.ViDeConv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.IrDetail = DetailFeatureExtraction()
        self.IrDeConv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        #en decode
        self.EnDecode1 = UpConv2dBnLeakyRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.EnDecode2 = UpConv2dBnLeakyRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.EnDecode3 = UpConv2dBnLeakyRelu(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.EnSigmoid = Conv2dSigmoid(in_channels=16)
        #ir decode
        self.IrDecode1 = UpConv2dBnLeakyRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.IrDecode2 = UpConv2dBnLeakyRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.IrDecode3 = UpConv2dBnLeakyRelu(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.IrSigmoid = Conv2dSigmoid(in_channels=16)
        #fu decode
        self.FuDecode1 = UpConv2dBnLeakyRelu(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.FuDecode2 = UpConv2dBnLeakyRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.FuDecode3 = UpConv2dBnLeakyRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.FuDecode4 = UpConv2dBnLeakyRelu(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.FuSigmoid = Conv2dSigmoid(in_channels=16)

    def forward(self, vi_o, ir_o):
        il = self.IL(vi_o)
        #vi_encode
        vi = self.ViEncode1(vi_o)
        vi = self.ViEncode2(vi)
        vi = self.ViEncode3(vi)
        vi = self.ViEncode4(vi)
        #ir_encode
        ir = self.IrEncode1(ir_o)
        ir = self.IrEncode2(ir)
        ir = self.IrEncode3(ir)
        ir = self.IrEncode4(ir)
        #cross attaction
        vi_down = self.ViConv(vi)
        vi_down = self.ViPool(vi_down)
        ir_down = self.IrConv(ir)
        ir_down = self.IrPool(ir_down)
        vi_out, ir_out = self.CSA(vi_down, ir_down)
        vi_out = self.ViUpConv(vi_out)
        ir_out = self.IrUpConv(ir_out)
        #attaction out
        vi_base = vi_out + vi
        ir_base = ir_out + ir
        #detail out
        vi_detail = self.ViDetail(vi)
        vi_detail = self.ViDeConv(vi_detail)
        ir_detail = self.IrDetail(ir)
        ir_detail = self.ViDeConv(ir_detail)
        #en_decode
        en  = self.EnDecode1(vi_base + vi_detail)
        en  = self.EnDecode2(en)
        en  = self.EnDecode3(en)
        en  = self.EnSigmoid(en)
        #fu_decode
        base_ = vi_base + ir_base
        detail_ = vi_detail + ir_detail
        fu  = self.FuDecode1(torch.cat((base_, detail_), dim=1))
        fu  = self.FuDecode2(fu)
        fu  = self.FuDecode3(fu)
        fu  = self.FuDecode4(fu)
        fu  = self.FuSigmoid(fu)
        #ir_decode
        ir  = self.IrDecode1(ir_base + ir_detail)
        ir  = self.IrDecode2(ir)
        ir  = self.IrDecode3(ir)
        ir  = self.IrSigmoid(ir)
        return il, en, ir, fu
        
class SimplifiedModel(nn.Module):
    def __init__(self):
        super(SimplifiedModel, self).__init__()
        #vi encode
        self.ViEncode1 = Conv2dBnLeakyRelu(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.ViEncode2 = Conv2dBnLeakyRelu(in_channels=16, out_channels = 32, kernel_size=3, stride=1, padding=1)
        self.ViEncode3 = Conv2dBnLeakyRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ViEncode4 = Conv2dBnLeakyRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #ir encode
        self.IrEncode1 = Conv2dBnLeakyRelu(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.IrEncode2 = Conv2dBnLeakyRelu(in_channels=16, out_channels = 32, kernel_size=3, stride=1, padding=1)
        self.IrEncode3 = Conv2dBnLeakyRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.IrEncode4 = Conv2dBnLeakyRelu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #crossattaction
        self.ViConv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=4, padding=1)
        self.ViPool  = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.IrConv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=4, padding=1)
        self.IrPool  = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.CSA     = CrossSelfAttention(hid_channels=256)
        self.ViUpConv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=4, padding=1, dilation=2, output_padding=1)
        self.IrUpConv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=4, padding=1, dilation=2, output_padding=1)
        #detailfeature
        self.ViDetail = DetailFeatureExtraction()
        self.ViDeConv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.IrDetail = DetailFeatureExtraction()
        self.IrDeConv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        #fu decode
        self.FuDecode1 = UpConv2dBnLeakyRelu(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.FuDecode2 = UpConv2dBnLeakyRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.FuDecode3 = UpConv2dBnLeakyRelu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.FuDecode4 = UpConv2dBnLeakyRelu(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.FuSigmoid = Conv2dSigmoid(in_channels=16)
    
    def forward(self, vi_o, ir_o):
        #vi_encode
        vi = self.ViEncode1(vi_o)
        vi = self.ViEncode2(vi)
        vi = self.ViEncode3(vi)
        vi = self.ViEncode4(vi)
        #ir_encode
        ir = self.IrEncode1(ir_o)
        ir = self.IrEncode2(ir)
        ir = self.IrEncode3(ir)
        ir = self.IrEncode4(ir)
        #cross attaction
        vi_down = self.ViConv(vi)
        vi_down = self.ViPool(vi_down)
        ir_down = self.IrConv(ir)
        ir_down = self.IrPool(ir_down)
        vi_out, ir_out = self.CSA(vi_down, ir_down)
        vi_out = self.ViUpConv(vi_out)
        ir_out = self.IrUpConv(ir_out)
        #attaction out
        vi_base = vi_out + vi
        ir_base = ir_out + ir
        #detail out
        vi_detail = self.ViDetail(vi)
        vi_detail = self.ViDeConv(vi_detail)
        ir_detail = self.IrDetail(ir)
        ir_detail = self.ViDeConv(ir_detail)
        #fu_decode
        base_ = vi_base + ir_base
        detail_ = vi_detail + ir_detail
        fu  = self.FuDecode1(torch.cat((base_, detail_), dim=1))
        fu  = self.FuDecode2(fu)
        fu  = self.FuDecode3(fu)
        fu  = self.FuDecode4(fu)
        fu  = self.FuSigmoid(fu)
        return fu
        
if __name__ == "__main__":
    network = SimplifiedModel().to("cuda")
    network.eval()
    print('==> Building model..')
    input1 = torch.randn(1, 1, 480, 640).to("cuda")
    input2 = torch.randn(1, 1, 480, 640).to("cuda")
    flops, params = profile(network, (input1, input2))
    print('flops: %.3f G, params: %.3f M' % (flops / 1e9, params / 1e6))
        

