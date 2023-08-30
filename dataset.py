import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.neighbors import NearestNeighbors
from network.utils import TransformerCV
from network.groupnet import group_config

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class KITTIDataset(data.Dataset):
    def __init__(self, data_path, seq):
        super().__init__()

        #protocol setting
        db_frames = {'00': range(0,3000), '02': range(0,3400), '05': range(0,1000), '06': range(0,600)}
        query_frames = {'00': range(3200, 4541), '02': range(3600, 4661), '05': range(1200,2751), '06': range(800,1101)}
        
        self.pos_threshold = 5   #ground truth threshold
        
        #preprocessor
        self.input_transform = input_transform()
        self.transformer = TransformerCV(group_config)
        self.pts_step = 5

        #root pathes
        bev_path = data_path + '/imgs/'
        lidar_path = data_path + '/velodyne/'

        #geometry positions
        poses = np.loadtxt(data_path+'/pose.txt')
        positions = np.hstack([poses[:,3].reshape(-1,1),  poses[:,11].reshape(-1,1)])

        self.db_positions = positions[db_frames[seq], :]
        self.query_positions = positions[query_frames[seq], :]

        self.num_db = len(db_frames[seq])

        #image pathes
        images = os.listdir(bev_path)
        images.sort()
        self.images = []
        for idx in db_frames[seq]:
            self.images.append(bev_path+images[idx])
        for idx in query_frames[seq]:
            self.images.append(bev_path+images[idx])     

        self.positives = None
        self.distances = None

    def transformImg(self, img):
        xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
        xs=xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        pts = np.hstack((xs,ys))
        img = img.permute(1,2,0).detach().numpy()
        transformed_imgs=self.transformer.transform(img,pts)
        data = self.transformer.postprocess_transformed_imgs(transformed_imgs)
        return data

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.input_transform(img)
        img*=255
        img = self.transformImg(img)
        
        return  img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_positions)

            self.distances, self.positives = knn.radius_neighbors(self.query_positions,
                    radius=self.pos_threshold)

        return self.positives
