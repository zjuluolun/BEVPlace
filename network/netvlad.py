import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """
 
    def __init__(self, norm=2, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps
 
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)
 
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
               
class NeighborAggregation(nn.Module):
    def __init__(self):
        super(NeighborAggregation, self).__init__()

    def forward(self, x):
        b, k, c = x.size()
        num_s = int(np.sqrt(k))
        x_2d = x.permute(0,2,1).view(b,c,num_s, num_s)

        y = torch.zeros(x_2d.size()).cuda()
        for i in range(num_s):
            for j in range(num_s):
                for k in (-1,2):
                    for l in (-1,2):
                        y[:,:,i,j] += x_2d[:,:,(i-k)%num_s,(j-l)%num_s]-x_2d[:,:,i,j]
                y /= 8

        # y = F.conv2d(x_2d, weight, padding=1)
        y = y.view(b,c,-1).permute(0,2,1)

        y = torch.cat((x,y),dim=2)
        return y


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out



class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=1,
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 with_shift=False,
                 **kwargs,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        self.with_shift = with_shift

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source,
                **kwargs,
                ):
        # source, target: [B, L, C]
        query, key, value = source, source, source

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message

class CenterAttention(nn.Module):
    def __init__(self):
        super(CenterAttention, self).__init__()

    def forward(self, x):
        y = x + x[:,x.size(1)//2+1,:].unsqueeze(1)
        # z = x[:,x.size(1)//2+1,:].unsqueeze(1).repeat(1,x.size(1),1)
        return x#torch.cat((z,y),axis=2)

class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, dim=128, num_clusters=256, lamb=2, groups=4, max_frames=400):
        super(NeXtVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.K = num_clusters
        self.G = groups
        self.group_size = int((lamb * dim) // self.G)

        # self.transformer_input = TransformerLayer(d_model=self.dim)
        self.transformer = TransformerLayer(d_model=self.K)
        # self.transformer2 = TransformerLayer(d_model=self.K)
        
        # expansion FC
        self.fc0 = nn.Linear(dim, lamb * dim)
        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * dim, self.G * self.K)
        # attention over groups FC
        self.fc_g = nn.Linear(lamb * dim, self.G)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.group_size, self.K))

        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x, mask=None):
        #         print(f"x: {x.shape}")
        x = x[:,:,:,0].permute(0,2,1)
        # x = self.transformer_input(x)
        _, M, N = x.shape
        # expansion FC: B x M x N -> B x M x λN
        x_dot = self.fc0(x)

        # reshape into groups: B x M x λN -> B x M x G x (λN/G)
        x_tilde = x_dot.reshape(-1, M, self.G, self.group_size)

        # residuals across groups and clusters: B x M x λN -> B x M x (G*K)
        WgkX = self.fc_gk(x_dot)
        # print(WgkX.size())
        WgkX = self.bn0(WgkX)

        # residuals reshape across clusters: B x M x (G*K) -> B x (M*G) x K
        WgkX = WgkX.reshape(-1, M * self.G, self.K)
        # WgkX = self.transformer(WgkX)
        # WgkX = self.transformer2(WgkX)
        # softmax over assignment: B x (M*G) x K -> B x (M*G) x K
        alpha_gk = F.softmax(WgkX, dim=-1)

        # attention across groups: B x M x λN -> B x M x G
        alpha_g = torch.sigmoid(self.fc_g(x_dot))
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))

        # reshape across time: B x M x G -> B x (M*G) x 1
        alpha_g = alpha_g.reshape(-1, M * self.G, 1)

        # apply attention: B x (M*G) x K (X) B x (M*G) x 1 -> B x (M*G) x K
        activation = torch.mul(alpha_gk, alpha_g)

        # sum over time and group: B x (M*G) x K -> B x 1 x K
        a_sum = torch.sum(activation, -2, keepdim=True)

        # calculate group centers: B x 1 x K (X) 1 x (λN/G) x K -> B x (λN/G) x K
        a = torch.mul(a_sum, self.cluster_weights2)

        # permute: B x (M*G) x K -> B x K x (M*G)
        activation = activation.permute(0, 2, 1)

        # reshape: B x M x G x (λN/G) -> B x (M*G) x (λN/G)
        reshaped_x_tilde = x_tilde.reshape(-1, M * self.G, self.group_size)

        # cluster activation: B x K x (M*G) (X) B x (M*G) x (λN/G) -> B x K x (λN/G)
        vlad = torch.matmul(activation, reshaped_x_tilde)
        # print(f"vlad: {vlad.shape}")

        # permute: B x K x (λN/G) (X) B x (λN/G) x K
        vlad = vlad.permute(0, 2, 1)
        # distance to centers: B x (λN/G) x K (-) B x (λN/G) x K
        vlad = torch.sub(vlad, a)
        # normalize: B x (λN/G) x K
        vlad = F.normalize(vlad, 1)
        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)
        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)

        return vlad

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

        self.wq = nn.Linear(self.dim, 64)
        self.wk = nn.Linear(self.dim, 64)

        self.mean = nn.Parameter(torch.ones(8, 1)/8)
        self.bias = nn.Parameter(torch.Tensor([1.0/8]))

        self.transformer = TransformerLayer(d_model=self.num_clusters)
        # self.wv = nn.Linear(self.dim, 64)
    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )
            

    def forward(self, x):
        N, C = x.shape[:2]

        # if self.normalize_input:
        #     x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        
        

        x_flatten = x.view(N, C, -1)

        
        
        '''
        reses = torch.zeros([N, self.num_clusters, x_flatten.size(-1)], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): 
            residual = x_flatten.unsqueeze(1) - self.centroids[C:C+1, :].unsqueeze(0).unsqueeze(-1)
            residual = torch.sum(residual**2,dim=2)
            reses[:, C, :] = residual
        
        reses_min = torch.min(reses, dim = 1)
        loss = reses_min
        '''

        

        # soft_assign = self.conv(x).view(N, self.num_clusters, -1)

        # soft_assign_attention = self.transformer(soft_assign.permute(0,2,1)).permute(0,2,1)

        # soft_assign = F.softmax(soft_assign_attention, dim=1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            # residual *= scores.unsqueeze(1).unsqueeze(1)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class GlobalAggregator(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self):
      
        super(GlobalAggregator, self).__init__()
        self.transformer = TransformerLayer(d_model=128)
        self.transformer1 = TransformerLayer(d_model=128)
        self.gem_pool = GeneralizedMeanPooling()

    def forward(self, x):
        N, C = x.shape[:2]

        
        x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x_flatten = x.view(N, C, -1).permute(0,2,1)
        
        # feature = self.transformer(x_flatten)
        # feature = self.transformer1(feature)
        # feature = F.softmax(feature, dim=2)

        vlad = self.gem_pool(x_flatten.permute(0,2,1).view(N,C,int(np.sqrt(x.shape[2])),int(np.sqrt(x.shape[2])))).squeeze()
        # print(vlad.size(),vlad[0,:])

        return vlad


class NetBoW(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=2048, dim=128, 
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetBoW, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x_flatten = x.view(N, C, -1)
        # soft-assignment
        bog = torch.zeros([N, self.num_clusters], dtype=x.dtype, layout=x.layout, device=x.device)
        for i in range(x_flatten.size()[2]):
        # for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = torch.mean(torch.abs(x_flatten[:,:,i].view(N,-1,1)-self.centroids.repeat(N,1,1).permute(0,2,1)),dim=1)
            # residual = torch.mean(residual,dim=1)
            # residual )
            # residual = F.relu(residual-torch.mean(residual))
            residual = F.softmax(-residual*1000,dim=1)
            #residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            bog += residual
        # bog[:,C] /= x_flatten.size()[2]
        # soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # soft_assign = F.softmax(soft_assign, dim=1)
        # bog = F.avg_pool1d(soft_assign, kernel_size=soft_assign.size()[-1])
        bog = F.normalize(bog, p = 2 ,dim=1).view(N, self.num_clusters)
        return bog

        # x_flatten = x.view(N, C, -1)
        
        # # calculate residuals to each clusters
        # vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        # for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
        #     residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
        #             self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        #     residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
        #     vlad[:,C:C+1,:] = residual.sum(dim=-1)

        # vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # vlad = vlad.view(x.size(0), -1)  # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        # return vlad
