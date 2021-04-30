import torch
import torch.nn as nn
import torch.nn.functional as F
from ..net_util import *


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        #print("ok5")
        #print(up1.shape)
        up1 = self._modules['b1_' + str(level)](up1)
        ##print("ok1")
        ##print(self._modules['b1_' + str(level)])(up1)
        #print("ok6")
        #print(up1.shape)
        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        #print("ok7")
        #print(low1.shape)
        low1 = self._modules['b2_' + str(level)](low1)
        ##print("ok2")
        ##print(self._modules['b2_' + str(level)])
        #print("ok8")
        #print(low1.shape)
        if level > 1:
            low2 = self._forward(level - 1, low1)
            #print("ok9")
            #print(low2.shape)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
            #print("ok3")
            ##print(self._modules['b2_plus_' + str(level)])
            #print("ok10")
            #print(low2.shape)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        ##print("ok4")
        ##print(self._modules['b3_' + str(level)])
        #print("ok11")
        #print(low3.shape)
        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        #print("ok12")
        #print(up2.shape)
       # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)
        ###print("ok3")
        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):
    def __init__(self, opt):
        super(HGFilter, self).__init__()
        self.num_modules = opt.num_stack

        self.opt = opt

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)#anche qua 

        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt.norm) #anche qua
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt.norm) #128x128 128 feature

        print(ConvBlock(128, 128, self.opt.norm))
        self.conv4 = ConvBlock(128, 256, self.opt.norm) #128x128 256 feature
        
        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt.norm))#cambia qua rispetto a pifuhd

            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.opt.norm))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            opt.hourglass_dim, kernel_size=1, stride=1, padding=0))#anche qua
            ##print("ok2")
            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim,
                                                                 256, kernel_size=1, stride=1, padding=0)) #anche qua

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        print("ok1")
        print(x.shape)
        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!') #anche qua

        normx = x
        print("ok2")
        print(x.shape)
        x = self.conv3(x)
        print("ok3")
        print(x.shape)
        x = self.conv4(x)
        print("ok4")
        print(x.shape)
        previous = x

        outputs = []
        for i in range(self.num_modules):
            ##print("ok0")
            hg = self._modules['m' + str(i)](previous)#usa hourglass qua
            ##print("ok1")

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)#qua non e' ourglass
            ##print("ok2")
            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)
         
            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, tmpx.detach(), normx
