import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import normalize_coordinates, interpolate_feats, to_cuda, dim_extend, l2_normalize

import os

class GroupNetConfig:
    def __init__(self):
        self.sample_scale_begin = 0
        self.sample_scale_inter = 0.5 
        self.sample_scale_num = 5 

        self.sample_rotate_begin = -90
        self.sample_rotate_inter = 45 
        self.sample_rotate_num = 5   

group_config = GroupNetConfig()

class VanillaLightCNN(nn.Module):
    def __init__(self):
        super(VanillaLightCNN, self).__init__()
        self.conv0=nn.Sequential(
            nn.Conv2d(3,16,5,1,2,bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16,32,5,1,2,bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )

        self.conv1=nn.Sequential(
            nn.Conv2d(32,32,5,1,2,bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,32,5,1,2,bias=False),
            nn.InstanceNorm2d(32),
        ) 

    def forward(self, x):
        x=self.conv1(self.conv0(x))
        x=l2_normalize(x,axis=1)
        return x

class ExtractorWrapper(nn.Module):
    def __init__(self,scale_num, rotation_num):
        super(ExtractorWrapper, self).__init__()
        self.extractor=VanillaLightCNN()
        self.sn, self.rn = scale_num, rotation_num 

    def forward(self,img_list,pts_list):
        '''

        :param img_list:  list of [b,3,h,w]
        :param pts_list:  list of [b,n,2]
        :return:gefeats [b,n,f,sn,rn]
        '''
        assert(len(img_list)==self.rn*self.sn)
        gfeats_list,neg_gfeats_list=[],[]
        # feature extraction
        for img_index,img in enumerate(img_list):
            # extract feature
            feats=self.extractor(img)
            gfeats_list.append(interpolate_feats(img,pts_list[img_index],feats)[:,:,:,None])
            

        gfeats_list=torch.cat(gfeats_list,3)  # b,n,f,sn*rn
        b,n,f,_=gfeats_list.shape
        gfeats_list=gfeats_list.reshape(b,n,f,self.sn,self.rn)
        
        return gfeats_list

class BilinearGCNN(nn.Module):
    def __init__(self):
        super(BilinearGCNN, self).__init__()
        self.network1_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network1_embed1_relu = nn.ReLU(True)

        self.network1_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network1_embed2_relu = nn.ReLU(True)

        self.network1_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 8, 3, 1, 1),
        )

        ###########################
        self.network2_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network2_embed1_relu = nn.ReLU(True)

        self.network2_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network2_embed2_relu = nn.ReLU(True)

        self.network2_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, 1, 1),
        )

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''
        b, n, f, ssn, srn = x.shape
        assert (ssn == 5 and srn == 5)
        x = x.reshape(b * n, f, ssn, srn)

        x1 = self.network1_embed1_relu(self.network1_embed1(x) + self.network1_embed1_short(x))
        x1 = self.network1_embed2_relu(self.network1_embed2(x1) + self.network1_embed2_short(x1))
        x1 = self.network1_embed3(x1)

        x2 = self.network2_embed1_relu(self.network2_embed1(x) + self.network2_embed1_short(x))
        x2 = self.network2_embed2_relu(self.network2_embed2(x2) + self.network2_embed2_short(x2))
        x2 = self.network2_embed3(x2)

        x1 = x1.reshape(b * n, 8, 25)
        x2 = x2.reshape(b * n, 16, 25).permute(0, 2, 1)  # b*n,25,16
        x = torch.bmm(x1, x2).reshape(b * n, 128)  # b*n,8,25
        assert (x.shape[1] == 128)
        x=x.reshape(b,n,128)
        x=l2_normalize(x,axis=2)
        return x

class EmbedderWrapper(nn.Module):
    def __init__(self):
        super(EmbedderWrapper, self).__init__()
        self.embedder=BilinearGCNN()

    def forward(self, gfeats):
        # group cnns
        gefeats=self.embedder(gfeats) # b,n,f
        return gefeats

class GroupNet(nn.Module):
    def __init__(self, config=group_config):
        super(GroupNet, self).__init__()
        self.scale_num = config.sample_scale_num
        self.rotation_num = config.sample_rotate_num

        self.extractor=ExtractorWrapper(self.scale_num, self.rotation_num).cuda()
        self.embedder=EmbedderWrapper().cuda()

    def forward(self, input):
        (img_list,pts_list) = input
        gfeats=self.extractor(dim_extend(img_list),dim_extend(pts_list))
        efeats=self.embedder(gfeats)
        return efeats
