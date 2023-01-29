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


import numpy as np
import torch.nn as nn

nc = 3
ndf = 64
stride = 2

def get_comtx(I):
    x = np.asarray(I)
    from skimage.feature import greycomatrix
    comtx = lambda y: greycomatrix(y, [5], [0], 256, symmetric=True, normed=True)

    return np.stack((comtx(x[:,:,0]),comtx(x[:,:,1]),comtx(x[:,:,2])),2).squeeze()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Nataraj2019(nn.Module):
    def __init__(self):
        super(Nataraj2019, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, 32, 3,stride=stride,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, bias=False),

            nn.MaxPool2d(kernel_size=3,stride=stride),

            nn.Conv2d(32, 64, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, bias=False),

            nn.MaxPool2d(kernel_size=3,stride=stride),

            nn.Conv2d(64, 128, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, bias=False),

            nn.MaxPool2d(kernel_size=3,stride=stride),

            Flatten(),
            nn.Linear(12800,256),
            nn.Linear(256, 2)

        #Questo lho cambiato io...output sono 2...
            #nn.Sigmoid()
        )

    def forward(self, input,labels=True):
        output = self.main(input)

        return output

