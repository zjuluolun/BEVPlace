import numpy as np 


def svdICP(src,dst):
    src = np.array(src).T
    dst = np.array(dst).T
    
    mean_src = np.mean(np.array(src),axis=1,keepdims=True)
    mean_dst = np.mean(np.array(dst),axis=1,keepdims=True)

    src_norm = src-mean_src
    dst_norm = dst-mean_dst

    mat_s = src_norm.dot(dst_norm.T)
    u, sigma, v_t = np.linalg.svd(mat_s)
    temp = u.dot(v_t)

    det = np.linalg.det(temp)
    s = np.array([[1,0],[0,det]])

    mat_r = v_t.T.dot(s).dot(u.T)

    translation = mean_dst.T - mean_src.T.dot(mat_r.T)
    return np.hstack((mat_r, translation.reshape(-1,1)))

def rigidRansac(points1, points2):
    points1 = points1[:,[1,0]]
    points2 = points2[:,[1,0]]
    max_cs_num = 0
    mask = np.zeros((points1.shape[0],1))
    mat = np.zeros((2,3))
    for i in range(1000):
        idx1 = np.random.randint(points1.shape[0])
        idx2 = np.random.randint(points1.shape[0])
        idx3 = np.random.randint(points1.shape[0])
        x = points1[[idx1,idx2], :]
        y = points2[[idx1,idx2], :]

        rot_mat = svdICP(x,y) 
        y_hat = points1.dot(rot_mat[:2,:2].T)+rot_mat[:2,2]
        err = np.abs(y_hat - points2)
        err = np.sqrt(np.sum(err**2, axis=1))

        consensus_num = np.sum(err<0.5)
        if consensus_num > max_cs_num:
            max_cs_num = consensus_num
            mask = err<0.5
            mat = rot_mat

    points1_c = points1[mask]
    points2_c = points2[mask]

    mat = svdICP(points1_c, points2_c)

    return mat , mask, max_cs_num
