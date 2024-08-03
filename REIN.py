import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torchvision.models as models

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))


    def init_params(self, clsts, traindescs):

        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

            

    def forward(self, x):
        N, C = x.shape[:2]
        x_flatten = x.view(N, C, -1)
        
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class REM(nn.Module):
    def __init__(self, from_scratch=False, rotations=8):
        super(REM, self).__init__()
        
        # cnn backbone
        pretrain = not from_scratch
        encoder = models.resnet34(pretrained=pretrain) #resnet34
        layers = list(encoder.children())[:-4]
        self.encoder = nn.Sequential(*layers)

        # rotations
        self.angles = -torch.arange(0,359.00001,360.0/rotations)/180*torch.pi

    
    def forward(self, x):
        
        equ_features = []
        
        batch_size = x.size(0)

        for i in range(len(self.angles)):

            # input warp grids
            aff = torch.zeros(batch_size,2,3).cuda()
            aff[:,0,0]=torch.cos(-self.angles[i])
            aff[:,0,1]=torch.sin(-self.angles[i])
            aff[:,1,0]=-torch.sin(-self.angles[i])
            aff[:,1,1]=torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, torch.Size(x.size()),align_corners=True).type(x.type())
            
            # input warp
            warped_im = F.grid_sample(x, grid,align_corners=True,mode='bicubic')
                                    
            # cnn backbone feature
            out = self.encoder(warped_im) 

            # output feature warp grids           
            if i==0:
                im1_init_size = out.size()

            aff = torch.zeros(batch_size,2,3).cuda()
            aff[:,0,0]=torch.cos(self.angles[i])
            aff[:,0,1]=torch.sin(self.angles[i])
            aff[:,1,0]=-torch.sin(self.angles[i])
            aff[:,1,1]=torch.cos(self.angles[i])
            grid = F.affine_grid(aff, torch.Size(im1_init_size),align_corners=True).type(x.type())

            # output feature warp    
            out = F.grid_sample(out, grid ,align_corners=True,mode='bicubic')

            equ_features.append(out.unsqueeze(-1))
        

        equ_features = torch.cat(equ_features, axis=-1)  # B C H W R

        B, C, H, W, R = equ_features.shape
        equ_features=torch.max(equ_features,dim=-1,keepdim=False)[0] # max pooling along rotations

        aff = torch.zeros(batch_size,2,3).cuda()
        aff[:,0,0]=1
        aff[:,0,1]=0
        aff[:,1,0]=0
        aff[:,1,1]=1

        
        # upsample for NetVLAD
        B,C,H,W = x.size()
        grid = F.affine_grid(aff, torch.Size((B, C, H//4, W//4)),align_corners=True).type(x.type())#,align_corners=True)
        out1 = F.grid_sample(equ_features, grid,align_corners=True,mode='bicubic')
        out1 = F.normalize(out1, dim=1)
        
        # upsample for keypoints
        grid = F.affine_grid(aff, torch.Size((B, C, H, W)),align_corners=True).type(x.type())#,align_corners=True)
        out2 = F.grid_sample(equ_features, grid,align_corners=True,mode='bicubic')
        out2 = F.normalize(out2, dim=1)
        
        return out1, out2

class REIN(nn.Module):
    def __init__(self):
        super(REIN, self).__init__()
        self.rem = REM()
        self.pooling = NetVLAD()

        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim*64
    
    def forward(self, x):

        out1, local_feats = self.rem(x)

        global_desc = self.pooling(out1)

        return out1, local_feats, global_desc
