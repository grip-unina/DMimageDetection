# code of https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet

import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9146867
#
# SRNet is a breakthrough in CNN-based steganalysis in that 
# it can be trained with randomly initialized parameters
# without any tricks. The authors found that it is crucial to adopt
# unpooled layers, which maintains the input dimension by disabling
# pooling and using stride 1, in the front part of the detector.
#
#

class Srnet(nn.Module):
    def __init__(self, padding=1):
        super(Srnet, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, 
                                kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=padding)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=padding)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=padding)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=padding)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,
            kernel_size=3, stride=1, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn122 = nn.BatchNorm2d(512)
        # avgp = torch.mean() in forward before fc
        # Fully Connected layer
        self.fc = nn.Linear(512*1*1, 2)

    def change_output(self, num_classes):
        self.fc = nn.Linear(512*1*1, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        return self
    
    def change_input(self, num_inputs):
        data = self.layer1.weight.data
        old_num_inputs = int(data.shape[1])
        if num_inputs>old_num_inputs:
            times = num_inputs//old_num_inputs
            if (times*old_num_inputs)<num_inputs:
                times = times+1
            data = data.repeat(1,times,1,1) / times
        elif num_inputs==old_num_inputs:
            return self
        
        data = data[:,:num_inputs,:,:]
        print(self.layer1.weight.data.shape, '->', data.shape)
        self.layer1.weight.data = data
        
        return self
    
    def features(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)
        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool3(bn)
        res = torch.add(convs, pool)
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool4(bn)
        res = torch.add(convs, pool)
        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)
        # print("L12:",res.shape)
        return bn
    
    def forward(self, inputs):
        bn = self.features(inputs)
        avgp = torch.mean(bn, dim=(2,3), keepdim=True)
        # fully connected
        flatten = avgp.view(avgp.size(0),-1)
        # print("flatten:", flatten.shape)
        fc = self.fc(flatten)
        # print("FC:",fc.shape)
        return fc
    
def srnet(pretrained=False, **kwargs):
    model = Srnet(**kwargs)
    if pretrained:
        import os
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'SRNet_model_weights.pt'))['model_state_dict'])
    return model