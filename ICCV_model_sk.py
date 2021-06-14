import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import torch.optim as optim


def init_net(net, init_type='orthogonal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

import torch
from torch import nn

# class TrainableEltwiseLayer(nn.Module)
#   def __init__(self, n, h, w):
#     super(TrainableEltwiseLayer, self).__init__()
# self.weights = nn.Parameter(torch.Tensor(1, n, h, w))  # define the
# trainable parameter

#   def forward(self, x):
#     # assuming x is of size b-1-h-w
#     return x * self.weights  # element-wise multiplication


class RefUnet(nn.Module):

    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=False)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=False)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=False)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=False)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=False)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=False)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=False)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


class SELayer(nn.Module):

    def __init__(self, ioc):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_reduction = nn.Conv2d(
            ioc * 2, ioc, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(ioc, 16, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(16, ioc, bias=False),
            nn.ReLU(inplace=False)
        )

    def forward(self, x1, x2):
        b, c1, _, _ = x1.size()
        b, c2, _, _ = x2.size()
        c = c1 + c2
        x = torch.cat([x1, x2], 1)
        # print(x.shape)
        x = self.channel_reduction(x)
        # print(x.shape)
        y = self.avg_pool(x).view(b, c1)
        y = self.fc(y)
        y = y.view(b, c1, 1, 1)

        return x * y.expand_as(x)


class SKConv(nn.Module):

    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2,
                          stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, DRE_out):
        '''
        input edr model
        output attention map mult on 2 branch
        '''
        fea_U = DRE_out
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat(
                    [attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)

        return attention_vectors


class Encoder(nn.Module):

    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.inc = in_channels
        self.l_conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.l_conv11 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.l_conv2 = nn.Conv2d(
            in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.l_conv22 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        # fusion feature's channel
        self.spaf1 = SPBlock(16, 16)
        self.spaf2 = SPBlock(16, 16)

    def forward(self, x1, x2):
        x1 = self.l_conv1(x1)
        x2 = self.l_conv2(x2)
        x1_att = self.spaf1(x1, x2)
        x2_att = self.spaf2(x2, x1)
        x1_out = self.l_conv11(x1_att)
        x2_out = self.l_conv22(x2_att)
        return x1_out,  x2_out


class Decoder(nn.Module):

    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.l_conv1 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.l_conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.l_conv3 = nn.Conv2d(
            16, in_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.l_conv1(x)
        x = self.l_conv2(x)
        x = self.l_conv3(x)

        return x


class SPBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(SPBlock, self).__init__()
        midplanes = int(outplanes // 2)

        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(
            3, 1), padding=(1, 0), bias=False)
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(
            1, 3), padding=(0, 1), bias=False)

        self.fuse_conv = nn.Conv2d(
            midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        self.conv_final = nn.Conv2d(
            midplanes, outplanes, kernel_size=1, bias=True)

        self.mask_conv_1 = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)
        # self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)

    def forward(self, input1, fusion):
        '''
        fusion feature attention
        (input + fusion feature) * attention mask
        '''
        x = fusion
        two_input = x + input1
        _, _, h, w = x.size()
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))
        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)

        hx = self.relu(self.fuse_conv(x_1_h + x_1_w))

        mask_1 = self.conv_final(hx).sigmoid()
        out1 = two_input * mask_1

        out = self.mask_relu(self.mask_conv_1(out1))

        # residual
        out += input1

        return out


class simple_model(nn.Module):

    def __init__(self):
        super(simple_model, self).__init__()
        self.enc1_y = Encoder(in_channels=1)
        self.dec_y = Decoder(in_channels=1)
        # self.se_y = SELayer(32)
        # self.se_cbcr = SELayer(32)
        self.enc1_cb = Encoder(in_channels=1)
        self.dec_cb = Decoder(in_channels=1)

        self.enc1_cr = Encoder(in_channels=1)
        self.dec_cr = Decoder(in_channels=1)
        # features, WH, M, G, r, stride=1 ,L=32
        self.sk = SKConv(32, 512, 2, 8, 2)
        # self.ref = RefUnet(3, 3)

    def forward(self, in1, in2):

        y1 = in1[:, 0:1, :, :]
        y2 = in2[:, 0:1, :, :]

        cb1 = in1[:, 1:2, :, :]
        cb2 = in2[:, 1:2, :, :]

        cr1 = in1[:, 2:3, :, :]
        cr2 = in2[:, 2:3, :, :]

        y1, y2 = self.enc1_y(y1, y2)
        mask_y = self.sk(y1 + y2)
        mask_y1 = mask_y[:, 0, :, :]
        mask_y2 = mask_y[:, 1, :, :]
        y1_att = y1 * mask_y1
        y2_att = y2 * mask_y2
        y = self.dec_y(y1_att + y2_att)

        cb1, cb2 = self.enc1_cb(cb1, cb2)
        mask_cb = self.sk(cb1 + cb2)
        mask_cb1 = mask_cb[:, 0, :, :]
        mask_cb2 = mask_cb[:, 1, :, :]
        cb1_att = cb1 * mask_cb1
        cb2_att = cb2 * mask_cb2
        cb = self.dec_cb(cb1_att + cb2_att)

        cr1, cr2 = self.enc1_cr(cr1, cr2)
        mask_cr = self.sk(cr1 + cr2)
        mask_cr1 = mask_cr[:, 0, :, :]
        mask_cr2 = mask_cr[:, 1, :, :]
        cr1_att = cr1 * mask_cr1
        cr2_att = cr2 * mask_cr2
        cr = self.dec_cr(cr1_att + cr2_att)

        cbcr = torch.cat([cb, cr], 1)
        return y, cbcr  # simple_output  # , output


class simple_model2(nn.Module):

    def __init__(self):
        super(simple_model2, self).__init__()
        self.enc1_y = Encoder(in_channels=1)
        self.dec_y = Decoder(in_channels=1)
        # self.se_y = SELayer(32)
        # self.se_cbcr = SELayer(32)
        self.enc1_cb = Encoder(in_channels=1)
        self.dec_cb = Decoder(in_channels=1)

        self.enc1_cr = Encoder(in_channels=1)
        self.dec_cr = Decoder(in_channels=1)
        # features, WH, M, G, r, stride=1 ,L=32
        self.sk = SKConv(32, 512, 2, 8, 2)

        # self.ref = RefUnet(3, 3)

    def forward(self, in1, in2):

        y1 = in1[:, 0:1, :, :]
        y2 = in2[:, 0:1, :, :]

        cb1 = in1[:, 1:2, :, :]
        cb2 = in2[:, 1:2, :, :]

        cr1 = in1[:, 2:3, :, :]
        cr2 = in2[:, 2:3, :, :]

        y1, y2 = self.enc1_y(y1, y2)
        mask_y = self.sk((y1 + y2))
        mask_y1 = mask_y[:, 0, :, :]
        mask_y2 = mask_y[:, 1, :, :]
        y1_att = y1 * mask_y1
        y2_att = y2 * mask_y2
        y = self.dec_y(y1_att + y2_att)

        cb1, cb2 = self.enc1_cb(cb1, cb2)
        mask_cb = self.sk((cb1 + cb2))
        mask_cb1 = mask_cb[:, 0, :, :]
        mask_cb2 = mask_cb[:, 1, :, :]
        cb1_att = cb1 * mask_cb1
        cb2_att = cb2 * mask_cb2
        cb = self.dec_cb(cb1_att + cb2_att)

        cr1, cr2 = self.enc1_cr(cr1, cr2)
        mask_cr = self.sk((cr1 + cr2))
        mask_cr1 = mask_cr[:, 0, :, :]
        mask_cr2 = mask_cr[:, 1, :, :]
        cr1_att = cr1 * mask_cr1
        cr2_att = cr2 * mask_cr2
        cr = self.dec_cr(cr1_att + cr2_att)

        ycbcr = torch.cat([y, cb, cr], 1)
        return ycbcr


class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
        self.block1 = simple_model2()
        # self.block2 = simple_model2()
        # self.block3 = simple_model2()

    def forward(self, in1, in2):
        out = self.block1(in1, in2)
        # out = self.block2(b1, b1) + b1
        # b2 = self.block2(in1, in2) + b1
        # out = self.block2(b1, b1) + b1
        # out = self.block3(b2, b2) + b2
        y, cbcr = out[:, 0:1, :, :], out[:, 1:3, :, :]
        return y, cbcr


# class model(nn.Module):

#     def __init__(self):
#         super(model, self).__init__()
#         self.block1 = simple_model2()
#         self.block2 = simple_model2()
#         self.block3 = simple_model2()

#     def forward(self, in1, in2):
#         b1 = self.block1(in1, in2)
#         out = self.block2(b1, b1) + b1
#         # b1 = self.block1(in1, in2)+ (in1 + in2) / 2
#         # out = self.block2(b1, b1) + b1
#         # b3 = self.block2(b2, b2)
#         y, cbcr = out[:, 0:1, :, :], out[:, 1:3, :, :]
#         return y, cbcr


# if __name__ =='__main__':
#     in1 = torch.ones(1,1,512,512).cuda()
#     in2 = torch.ones(1,1,512,512).cuda()
#     in3 = torch.ones(1,2,512,512).cuda()
#     in4 = torch.ones(1,2,512,512).cuda()
#     bas = simple_model().cuda()
#     print(bas)
#     print(bas(in1,in2,in3,in4))
#     y, cbcr = bas(in1,in2,in3,in4)

#     # skip = net(in1,in2)
# #     print(skip['fir_pool1'])
#     print(bas(in1,in2,in3,in4)[0].shape)
#     print(bas(in1,in2,in3,in4)[1].shape)
