import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data

import h5py

import faiss
from RANSAC import rigidRansac

kitti_seq_split_points = {"00":3000, "02":3400, "05":1000, "06":600, '08':1000}

class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path = './datasets/KITTI/',sample_inteval=1):
        super().__init__()

        # bev path
        imgs_p = os.listdir(dataset_path+seq+'/bev_imgs/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/bev_imgs/'+imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]

        # gt_pose
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')[::sample_inteval]


    def __getitem__(self, index):
        
        img = cv2.imread(self.imgs_path[index], 0)
        if 0:  #test rotation
            mat = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2 ), np.random.randint(0,360), 1)
            img = cv2.warpAffine(img, mat, img.shape[:2])

        img = (img.astype(np.float32))/256 
        img = img[np.newaxis, :, :].repeat(3,0)
        
        return  img, index

    def __len__(self):
        return len(self.imgs_path)


def evaluateResults(seq, global_descs, local_feats, dataset, match_results_save_path=None):

    if match_results_save_path is not None: 
        os.system('mkdir -p ' + match_results_save_path)
        all_errs = []
        local_feats = local_feats.transpose(0,2,3,1)

    gt_thres = 5  # gt threshold
    faiss_index = faiss.IndexFlatL2(global_descs.shape[1]) 
    faiss_index.add(global_descs[:kitti_seq_split_points[seq]])

    _, predictions = faiss_index.search(global_descs[kitti_seq_split_points[seq]+200:], 1)  #top1
    
    
    eval_start_split_point = kitti_seq_split_points[seq]+200  
    all_positives = 0
    tp = 0
    for q_idx, pred in enumerate(predictions):

        query_idx = eval_start_split_point+q_idx
        gt_dis = (dataset.poses[query_idx] - dataset.poses[:kitti_seq_split_points[seq]])**2
        positives = np.where(np.sum(gt_dis[:,[3,7,11]],axis=1) < gt_thres**2 )[0]
        if len(positives)>0:
            all_positives+=1
            if pred[0] in positives:
                tp += 1

            if match_results_save_path is not None:

                index = pred[0]


                query_im = dataset[query_idx][0].transpose(1,2,0)*256
                db_im = dataset[index][0].transpose(1,2,0)*256

                query_im = query_im.astype(np.uint8)
                db_im = db_im.astype(np.uint8)

                fast = cv2.FastFeatureDetector_create()
                im_side = db_im.shape[0]

                query_kps = fast.detect(query_im, None)
                db_kps = fast.detect(db_im, None)

                
                query_des = [local_feats[query_idx][int(kp.pt[1]),int(kp.pt[0])] for kp in query_kps]
                db_des = [local_feats[index][int(kp.pt[1]),int(kp.pt[0])] for kp in db_kps]
                
                query_des = np.array(query_des)
                db_des = np.array(db_des)
                
                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(query_des, db_des, k=2)
                
                

                all_match = [m[0] for m in matches]
                points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match]) 
                points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

                H, mask, max_csc_num = rigidRansac((np.array([[im_side//2,im_side//2]]-points1)*0.4),(np.array([[im_side//2,im_side//2]]-points2))*0.4)# cv2.findHomography(points1, points2, cv2.RANSAC, 4.0)
                
                q_pose = dataset.poses[query_idx]

                q_pose = np.hstack((q_pose[:12].reshape(3,4)[:2,:2], q_pose[:12].reshape(3,4)[:2,3].reshape(-1,1)))
                q_pose = np.vstack((q_pose,np.array([[0,0,1]])))

                db_pose = dataset.poses[index]
                db_pose = np.hstack((db_pose[:12].reshape(3,4)[:2,:2], db_pose[:12].reshape(3,4)[:2,3].reshape(-1,1)))
                db_pose = np.vstack((db_pose,np.array([[0,0,1]])))

                relative_gt = np.linalg.inv(db_pose).dot((q_pose))
                relative_H = np.vstack((H, np.array([[0,0,1]])))
                
                err = np.linalg.inv(relative_H).dot(relative_gt)
                err_theta = np.abs(np.arctan2(err[0,1], err[0,0])/np.pi*180)
                err_trans = np.sqrt(err[0,2]**2+err[1,2]**2)

                if err_theta>5 or err_trans>2:
                    print('bug')
                all_errs.append([err_trans, err_theta])
                              
                good_match = [all_match[i] for i in range(len(mask)) if  mask[i]]
                db_im = db_im*3
                db_im[:,:,:2]=0


                im = cv2.drawMatches(query_im.astype(np.uint8), query_kps, db_im.astype(np.uint8), db_kps, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                out_im = np.zeros((im.shape[0]*2, db_im.shape[1]*3,3))
                out_im[:im.shape[0], :db_im.shape[1]] = query_im
                out_im[:im.shape[0], db_im.shape[1]:db_im.shape[1]*2] = db_im
                out_im[:im.shape[0], db_im.shape[1]*2:] = db_im+query_im

                out_im[-im.shape[0]:, :db_im.shape[1]*2] = im
                

                H = relative_H 
                mat = cv2.getRotationMatrix2D((query_im.shape[0]//2, query_im.shape[0]//2), np.arctan2(-H[0,1], H[0,0])/np.pi*180, 1.0)
                mat[0,2] -= H[1,2]/0.4
                mat[1,2] -= H[0,2]/0.4
                mat = np.vstack((mat,np.array([[0,0,1]])))
                mat = np.linalg.inv(mat)[:2,:]
                im_warp = cv2.warpAffine(db_im, mat, query_im.shape[:2])

                im_warp[:,:,:2]=0
                out_im[-im.shape[0]:, db_im.shape[1]*2:db_im.shape[1]*3] = im_warp+query_im                
                cv2.imwrite(match_results_save_path+str(1000000+query_idx)[1:]+".png", out_im)

    
      
    recall_top1 = tp / all_positives #tp/(tp+fp)

    

    if match_results_save_path is not None:
        all_errs = np.array(all_errs)
        success_loc = (all_errs[:,0]<2) & (all_errs[:,1]<5)
        success_rate = np.sum(success_loc)/all_positives
        mean_trans_err = np.mean(all_errs[success_loc,1])
        mean_rot_err = np.mean(all_errs[success_loc,0]) 
        return recall_top1, success_rate, mean_trans_err, mean_rot_err
    else:
        return recall_top1

        
def collate_fn(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    
    negatives = torch.cat(negatives, 0)
    indices = list(indices)

    return query, positive, negatives, indices


class TrainingDataset(data.Dataset):
    def __init__(self, dataset_path = './datasets/KITTI/',seq='00'):
        super().__init__()

        # bev path
        imgs_p = os.listdir(dataset_path+seq+'/bev_imgs/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/bev_imgs/'+i for i in imgs_p]

        # gt_pose, only first 3000 frames of KITTI for training
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')
        self.poses = self.poses[:3000]
        
        # neg, pos threshold
        self.pos_thres = 5
        self.neg_thres = 7 # 

        # compute pos and negs for each query
        self.num_neg = 10
        self.positives = []
        self.negatives = []
        for qi in range(len(self.poses)):
            q_pose = self.poses[qi]
            dises = np.sqrt(np.sum(((q_pose-self.poses)**2)[:,[3,7,11]],axis=1))            
            indexes = np.argsort(dises)

            remap_index = indexes[np.where(dises[indexes]<self.pos_thres)[0]]
            self.positives.append(remap_index)
            self.positives[-1] = self.positives[-1][1:] #exclude query itself

            negs = indexes[np.where(dises[indexes]>self.neg_thres)[0]]
            self.negatives.append(negs)

        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images



    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            # pos_idx = self.positives[index][min_idx]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            pos_idx = self.positives[index][0]
        
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        query = cv2.imread(self.imgs_path[index])
        # rot augmentation
        mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2 ), np.random.randint(0,360), 1)
        query = cv2.warpAffine(query, mat, query.shape[:2])
        
        query = query.transpose(2,0,1)


        positive = cv2.imread(join(self.imgs_path[pos_idx]))#           
        mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(0,360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2,0,1)
        

    
        query = (query.astype(np.float32))/256
        positive = (positive.astype(np.float32)/256)

        negatives = []

        for neg_i in neg_idx:
        
            negative = cv2.imread(self.imgs_path[neg_i])
            mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(0,360), 1)
            negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
            negative = negative.transpose(2,0,1)
            negative = (negative)/256
            
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)
