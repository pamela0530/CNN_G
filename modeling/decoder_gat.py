import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        # self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.5),
        #                                # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.1),
        #                                nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        self.last_conv2d = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.pool_gat = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.gatt= GAT(256, 100, 21, 0.6, 0.2, 1)
        self.num_class = num_classes

        self._init_weight()


    def forward(self, x, low_level_feat):

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(x)
        x_1 = self.last_conv2d(x)
        x =self.pool_gat(x)

        x_0 = x[0,:,:,:]
        h = x.size()[-2]
        w = x.size()[-1]
        x_0 = x_0.view(x.size(1), x.size()[-2] * x.size()[-1])
        x_0 = x_0.permute(1, 0)
        print(x_0.size())
        x_0 = self.gatt(x_0)
        x_0 = x_0.view(1, self.num_class, h, w)
        if x.size()[0]>1:
            print(x.size())
            x_1 = x[1, :, :, :]
            x_1 = x_1.view(x.size(1), x.size()[-2] * x.size()[-1])
            x_1 = x_1.permute(1, 0)
            print(x_1.size(),x_0.size())
            x_0 = torch.cat([x_0,self.gatt(x_1)],dim=0)


        #
        # x = x.view(x.size(1), x.size()[-2] * x.size()[-1])
        # x = x.permute( 1, 0)
        # print(x.size())
        # x = self.gatt(x)
        # x = x.view(1,self.num_class, h, w)
        x = F.interpolate(x_0, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = x_1 + x
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def build_Decoder_gat(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)