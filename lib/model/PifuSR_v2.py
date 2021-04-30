'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

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

class ResBlock(nn.Module):
    """
    Basic residual block for AMRSR.

    Parameters
    ---
    n_filters : int, optional
        a number of filters.
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x

class PifuSR_v2(nn.Module):
    def __init__(self, opt,n_blocks=2):
        super(PifuSR_v2, self).__init__()

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
            *[ResBlock(32) for _ in range(n_blocks)],
        )
        self.tail1=nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            #nn.LeakyReLU(0.2, True)

        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)

        )
        self.body2 = nn.Sequential(
            *[ResBlock(64) for _ in range(n_blocks)],
        )
        self.tail2=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.LeakyReLU(0.2, True)

        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)

        )
        self.body3 = nn.Sequential(
            *[ResBlock(128) for _ in range(n_blocks)],
        )
        self.tail3=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.LeakyReLU(0.2, True)


            
        )

        self.bottleneck=nn.Sequential(

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.bott2=nn.Sequential(

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        #valutare se andare piu giu
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
            #checco se devo usare softmax
            )

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)
    
    def forward(self, x):
        ##print("input")
        ##print(x.shape)
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

        bo=self.bottleneck(d3_f) #non sono sicuro qua

        new1=torch.cat((d3_f,bo),0) #concateno 64x64x128 con 64x64x128= 64x64x512

        up1_1=self.bott2(new1) #64x64x512
       
        up1=self.pixel_shuffle(up1_1) #128x128x128
        #vedo se passare up1 a 
        new2=torch.cat((d2_f,up1),0) #128x128x256

        up2_2=self.ups2(new2) #128x128x256
        up2=self.pixel_shuffle(up2_2) #256x256x64 sbaglio qualcosa dovrebbe essere 128

        new3=torch.cat((d1_f,up2),0) #256x256x128
        up3_2=self.ups3(new2)
        up3=self.pixel_shuffle(up3_2) #512x512x32

        fin=torch.cat((h,up3),0) #512x512x64 
        new_fin=self.ups4(fin) #layer_hr

        #volendo posso abbassare tutto di un channel, quindi partire da 32 al posto di 64 perche sono a 128x128x512


        img_SR=self.last(new_fin)

        
        return img_SR, layer_lr, layer_hr
    