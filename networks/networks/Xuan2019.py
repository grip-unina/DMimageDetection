#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import torch.nn as nn
nc = 3
ndf = 64

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 2, 4, 1, 0, bias=False),
            Flatten(),
            nn.Linear(2*5*5,2)
            #Questo lho cambiato io...output sono 2...
            #nn.Sigmoid()
        )

    def forward(self, input,labels=True):
        output = self.main(input)

        return output


#import numpy as np
#def gaussian_noise(image,mean,sigma):
#    image = np.asarray(image,dtype=np.float32)
#    row, col,ch = image.shape
#    gauss = np.random.normal(mean, sigma, (row, col, ch))
#    gauss = gauss.reshape(row, col, ch)
#    noisy = image + gauss
#    return noisy
#
#from PIL import ImageFilter
#def gaussian_blur(image,kernel):
#    return image.filter(ImageFilter.GaussianBlur(radius=kernel))

