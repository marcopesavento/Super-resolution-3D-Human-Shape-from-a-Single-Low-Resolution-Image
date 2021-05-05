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
from .common import *

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
            *[ResBlock(32) for _ in range(n_blocks)]
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
            *[ResBlock(64) for _ in range(n_blocks)]
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
            *[ResBlock(128) for _ in range(n_blocks)],
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

        self.upsample = nn.Upsample(scale_factor=opt.scale,
                                    mode='bicubic', align_corners=False)
    
    def forward(self, x):
        ##print("input")
        ##print(x.shape)
        h = self.upsample(x)
        
        h=self.head(h) #512x512x32  
        #print(h)
        
        

        ##print("upsample")
        ##print(h.shape)
        d1=self.down1(h)
       
        if self.residual:
            d1=self.body1(d1)
        d1_f=self.tail1(d1) #256x256x64 perche e a 32?
        ##print("down1")
        #print(d1_f)

        d2=self.down2(d1_f)
        if self.residual:
            d2=self.body2(d2)
        d2_f=self.tail2(d2)#128x128x128
        ##print("down2")
        #print(d2_f) #fin qui abbastanza ragionevoli
        d3=self.down3(d2_f) #64x64x128
        #print(d3)
        #da qua mi parte
        if self.residual:
            ##print("ok")
            d3=self.body3(d3)
            #print(d3)
        d3_f=self.tail3(d3) 
        ##print("down3")
        #print(d3_f)
        bo=self.bottleneck(d3_f) #non sono sicuro qua
        ##print("bott")
        #print(bo.shape)
        new1=torch.cat((d3_f,bo),1) #concateno 64x64x128 con 64x64x128= 64x64x512
        #print("bott2")
        #print(new1.shape)
        up1_1=self.bott2(new1) #64x64x512
        #print("up1_1")
        #print(up1_1.shape)
        up1=self.pixel_shuffle(up1_1) #128x128x128
        #print("up1")
        #print(up1.shape)
        #vedo se passare up1 a 
        new2=torch.cat((d2_f,up1),1) #128x128x256
        #print("new2")
        #print(new2.shape)

        up2_2=self.ups2(new2) #128x128x256
        #print("up2_2")
        #print(up2_2.shape)
        up2=self.pixel_shuffle(up2_2) #256x256x64 sbaglio qualcosa dovrebbe essere 128
        #print("up2")
        #print(up2.shape)
        new3=torch.cat((d1_f,up2),1) #256x256x128
        #print("new3")
        #print(new3.shape)
        up3_2=self.ups3(new3)
        #print("up3_2")
        #print(up3_2.shape)
        up3=self.pixel_shuffle(up3_2) #512x512x32
        #print("up3")
        #print(up3.shape)
        fin=torch.cat((h,up3),1) #512x512x64 
        ##print("fin")
        ##print(fin.shape)
        new_fin=self.ups4(fin) #layer_hr
        ##print("new_fin")
        ##print(new_fin.shape) 

        #volendo posso abbassare tutto di un channel, quindi partire da 32 al posto di 64 perche sono a 128x128x512


        img_SR=self.last(new_fin) #non son sicuro
        ##print("img")
        ##print(img_SR.shape)
        #print(img_SR)
        
        return img_SR, new2, new_fin
    