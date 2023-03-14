import numpy as np
import matplotlib.pyplot as plt
import pcl
import cv2
import os


parser = argparse.ArgumentParser(description='BEVPlace-Gen-BEV-Images')
parser.add_argument('--seq_path', type=str, default="./KITTI05/", help='path to data')

def getBEV(all_points): #N*3
    
    all_points_pc = pcl.PointCloud()
    all_points_pc.from_array(all_points)
    f = all_points_pc.make_voxel_grid_filter()
    
    ls = 0.4
    
    f.set_leaf_size(ls, ls, ls)
    all_points_pc=f.filter()
    all_points = np.array(all_points_pc.to_list())

    x_min = -40
    y_min = -40
    x_max = 40 
    y_max = 40

    x_min_ind = int(x_min/0.4)
    x_max_ind = int(x_max/0.4)
    y_min_ind = int(y_min/0.4)
    y_max_ind = int(y_max/0.4)

    x_num = x_max_ind-x_min_ind+1
    y_num = y_max_ind-y_min_ind+1

    mat_global_image = np.zeros(( y_num,x_num),dtype=np.uint8)
          
    for i in range(all_points.shape[0]):
        x_ind = x_max_ind-int(all_points[i,1]/0.4)
        y_ind = y_max_ind-int(all_points[i,0]/0.4)
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
    bins_path = os.listdir(args.seq_path+"/velodyne/")
    bins_path.sort()

    for i in range(len(bins_path)):

        b_p = bins_path[i]
        pcs = np.fromfile(args.seq_path+"/velodyne/"+'/'+b_p,dtype=np.float32).reshape(-1,4)[:,:3]

        pcs = pcs[np.where(np.abs(pcs[:,0])<25)[0],:]
        pcs = pcs[np.where(np.abs(pcs[:,1])<25)[0],:]
        pcs = pcs[np.where(np.abs(pcs[:,2])<25)[0],:]

        pcs = pcs.astype(np.float32)
        img, _, _ = getBEV(pcs)

        cv2.imwrite(args.seq_path+"/imgs/"+b_p[:-4]+".png",img)

exit()
