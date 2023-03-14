import torch.nn as nn

from network.netvlad import NetVLAD
from network.groupnet import GroupNet
from network.utils import to_cuda



class BEVPlace(nn.Module):
    def __init__(self):
        super(BEVPlace, self).__init__()
        self.encoder = GroupNet()
        self.netvlad = NetVLAD()

    def forward(self, input):
        input = to_cuda(input)
        local_feature = self.encoder(input) 
        local_feature = local_feature.permute(0,2,1).unsqueeze(-1)
        global_feature = self.netvlad(local_feature) 

        return global_feature
