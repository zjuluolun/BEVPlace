import numpy as np
import matplotlib.pyplot as plt
# import pcl
import open3d as o3d
import cv2
import os
import argparse
from tqdm import trange
parser = argparse.ArgumentParser(description='BEVPlace-Gen-BEV-Images')
parser.add_argument('--vel_path', type=str, default="/mnt/share_disk/KITTI/dataset/sequences/00/velodyne/", help='path to data')
parser.add_argument('--bev_save_path', type=str, default="./KITTI_new_imgs/00/imgs/", help='path to data')

def getBEV(all_points): #N*3
    
    all_points_pc = o3d.geometry.PointCloud()# pcl.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)#all_points_pc.from_array(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=0.4) #f = all_points_pc.make_voxel_grid_filter()
    

    all_points = np.asarray(all_points_pc.points)# np.array(all_points_pc.to_list())


    x_min = -40
    y_min = -40
    x_max = 40 
    y_max = 40

    x_min_ind = np.floor(x_min/0.4).astype(int)
    x_max_ind = np.floor(x_max/0.4).astype(int)
    y_min_ind = np.floor(y_min/0.4).astype(int)
    y_max_ind = np.floor(y_max/0.4).astype(int)

    x_num = x_max_ind-x_min_ind+1
    y_num = y_max_ind-y_min_ind+1

    mat_global_image = np.zeros(( y_num,x_num),dtype=np.uint8)
          
    for i in range(all_points.shape[0]):
        x_ind = x_max_ind-np.floor(all_points[i,1]/0.4).astype(int)
        y_ind = y_max_ind-np.floor(all_points[i,0]/0.4).astype(int)
        if(x_ind>=x_num or y_ind>=y_num):
            continue
        if mat_global_image[ y_ind,x_ind]<10:
            mat_global_image[ y_ind,x_ind] += 1

    max_pixel = np.max(np.max(mat_global_image))

    mat_global_image[mat_global_image<=1] = 0  
    mat_global_image = mat_global_image*10
    
    mat_global_image[np.where(mat_global_image>255)]=255
    mat_global_image = mat_global_image/np.max(mat_global_image)*255

    return mat_global_image,x_max_ind,y_max_ind


if __name__ == "__main__":

    args = parser.parse_args()
    bins_path = os.listdir(args.vel_path)
    bins_path.sort()
    os.system('mkdir -p '+args.bev_save_path)
    for i in trange(len(bins_path)):

        b_p = bins_path[i]
        pcs = np.fromfile(args.vel_path+'/'+b_p,dtype=np.float32).reshape(-1,4)[:,:3]

        # ang = np.random.randint(360)/180.0*np.pi
        # rot_mat = np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])
        # pcs = pcs.dot(rot_mat)

        pcs = pcs[np.where(np.abs(pcs[:,0])<25)[0],:]
        pcs = pcs[np.where(np.abs(pcs[:,1])<25)[0],:]
        pcs = pcs[np.where((np.abs(pcs[:,2])<25))[0],:]

        pcs = pcs.astype(np.float32)
        img, _, _ = getBEV(pcs)

        cv2.imwrite(args.bev_save_path+'/'+b_p[:-4]+".png",img)

exit()
