'''
MIT License

Copyright (c) 2022 Marco Pesavento, Marco Volino and Adrian Hilton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from ..net_util import *
from .common import *

class SuRSSR_v3(nn.Module):
    def __init__(self, opt,n_blocks=[], conv=default_conv):
        super(SuRSSR_v3, self).__init__()

        self.n_blocks=opt.n_block
        print((self.n_blocks))

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(opt.rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

        self.opt = opt
        self.residual = opt.residual
        #self.phase = opt.downsize #default 5
        self.scale = opt.scale
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)

        )
        self.body1 = nn.Sequential(
            *[ResBlock(conv,32,3) for _ in range(self.n_blocks[0])]
        )
        self.tail1=nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)

        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)

        )
        self.body2 = nn.Sequential(
            *[ResBlock(conv,64,3) for _ in range(self.n_blocks[1])]
        )
        self.tail2=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)

        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)

        )
        self.body3 = nn.Sequential(
            *[ResBlock(conv,128,3) for _ in range(self.n_blocks[2])],
        )
        self.tail3=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)


            
        )

        self.bottleneck=nn.Sequential(

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.bott2=nn.Sequential(

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.pixel_shuffle=nn.Sequential(

            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )
        
        self.ups2=nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            
            )
        self.ups3=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            
            )
        
        self.ups4=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            
            )

        self.last=nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
            )

        self.upsample = nn.Upsample(scale_factor=opt.scale,
                                    mode='bicubic', align_corners=False)
    
    def forward(self, x):
        h = self.upsample(x)
        h=self.head(h) #512x512x32  
        d1=self.down1(h)
     
        if self.residual:
            d1=self.body1(d1)
        d1_f=self.tail1(d1) #256x256x64 
        
        

        d2=self.down2(d1_f)
        if self.residual:
            d2=self.body2(d2)
        d2_f=self.tail2(d2)#128x128x128
        d3=self.down3(d2_f) #64x64x128
        
        if self.residual:
            d3=self.body3(d3)
        d3_f=self.tail3(d3) 
       
        bo=self.bottleneck(d3_f) 
        new1=torch.cat((d3_f,bo),1) 
        up1_1=self.bott2(new1) #64x64x512
        up1=self.pixel_shuffle(up1_1) #128x128x128
        new2=torch.cat((d2_f,up1),1) #128x128x256

        up2_2=self.ups2(new2) #128x128x256
        up2=self.pixel_shuffle(up2_2) #256x256x64 
        new3=torch.cat((d1_f,up2),1) #256x256x128
        up3_2=self.ups3(new3)
        up3=self.pixel_shuffle(up3_2) #512x512x32
        
        fin=torch.cat((h,up3),1) #512x512x64 
        new_fin=self.ups4(fin) 
        img_SR=self.last(new_fin) 

        
        return img_SR, new2, new_fin
    