import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.pyGAT import GAT
from modeling.GATlayers import GraphAttentionLayer
import numpy as np
np.set_printoptions(threshold=np.inf)


class Deeplab_GAT(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(Deeplab_GAT, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        for p in self.parameters():
            p.requires_grad = False
        self.gatt = GAT(256, 256, num_classes, 0.6, 0.2, 1)
        self.num_class = num_classes


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x_1 = self.decoder(x, low_level_feat)
        x_1 = F.interpolate(x_1, size=input.size()[2:], mode='bilinear', align_corners=True)
        # print(x.size())
        h = x.size()[-2]
        w = x.size()[-1]
        x_0 = x[0].unsqueeze(0)

        if not self.training:
            x_0, img_att = self.gatt(x_0)
            x_0 = F.interpolate(x_0, size=input.size()[2:], mode='bilinear', align_corners=True)
            return x_0, x_1, img_att, (w, h)
        else:
            x_0 = self.gatt(x_0)
            x_0 = F.interpolate(x_0, size=input.size()[2:], mode='bilinear', align_corners=True)

        # x_0 = self.gatt(x_0)
        # x_0 = F.interpolate(x_0, size=input.size()[2:], mode='bilinear', align_corners=True)


        if x.size()[0]>1:
            for i in range(x.size()[0]-1):
                x_i = x[i, :, :, :].unsqueeze(0)
                # x_i = x_i.view(x.size(1), x.size()[-2] * x.size()[-1])
                # x_i = x_i.permute(1, 0)
                # print(x_i.size())
                # if not self.training:
                #     x_i,img_att_2 = self.gatt(x_i)
                #     x_0 = torch.cat(
                #         [x_0, F.interpolate(x_i, size=input.size()[2:], mode='bilinear', align_corners=True)], dim=0)
                #     return x_0,x_1,img_att,(w,h)
                # else:
                #     x_i = self.gatt(x_i)
                #     x_0 = torch.cat(
                #         [x_0, F.interpolate(x_i, size=input.size()[2:], mode='bilinear', align_corners=True)], dim=0)
                x_i = self.gatt(x_i)
                x_0 = torch.cat(
                    [x_0, F.interpolate(x_i, size=input.size()[2:], mode='bilinear', align_corners=True)], dim=0)
        # x_0=x_0+x_1
        # print(x_0.data,x_1.data)


        return x_0,x_1,[],(w,h)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_100x_lr_params(self):
        modules = [self.gatt]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                # print(m)
                if isinstance(m[1], nn.Conv1d) or isinstance(m[1], GraphAttentionLayer):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            # print(p)
                            yield p


    def set_grad(self, para_on=False):
        grad_set_1 = [SynchronizedBatchNorm2d, nn.Conv2d, nn.BatchNorm2d ]

        modules = [self.backbone,self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        p.requires_grad = para_on
        modules_1 = [self.gatt]
        for i in range(len(modules_1)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv1d) or isinstance(m[1], GraphAttentionLayer):
                    for p in m[1].parameters():
                        if para_on:
                            p.requires_grad = False
                        else:
                            p.requires_grad = True




if __name__ == "__main__":
    model = Deeplab_GAT(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


