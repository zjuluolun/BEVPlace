from __future__ import print_function
import argparse
from os.path import join, isfile
from os import environ
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import faiss
from network.bevplace import BEVPlace
from network.utils import to_cuda

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='BEVPlace')
parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=40, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='checkpoints/checkpoint_custom_bev.pth.tar', help='Path to load checkpoint from, for resuming training or testing.')


def evaluate(eval_set, model):
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False, 
                pin_memory=cuda)

    model.eval()

    global_features = []
    with torch.no_grad():
        print('====> Extracting Features')
        with tqdm(total=len(test_data_loader)) as t:
            for iteration, (input, indices) in enumerate(test_data_loader, 1):
                if cuda:
                    input = to_cuda(input)
                batch_feature = model(input)
                global_features.append(batch_feature.detach().cpu().numpy())
                t.update(1)

    global_features = np.vstack(global_features)

    query_feat = global_features[eval_set.num_db:].astype('float32')
    db_feat = global_features[:eval_set.num_db].astype('float32')

    # print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(query_feat.shape[1])
    faiss_index.add(db_feat)

    # print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(query_feat, max(n_values)) 

    gt = eval_set.getPositives() 

    correct_at_n = np.zeros(len(n_values))
    whole_test_size = 0

    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) ==0 : 
            continue
        whole_test_size+=1
        for i,n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / whole_test_size
<<<<<<< Updated upstream
    print("tp+fn=%d"%(whole_test_size))
=======
    # print("tp+fn=%d"%(whole_test_size))
>>>>>>> Stashed changes
    recalls = {} 
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
    #     print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return recalls

import dataset as dataset

from network import netvlad

if __name__ == "__main__":
    opt = parser.parse_args()

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
     
    print('===> Building model')
    model = BEVPlace()
<<<<<<< Updated upstream
    resume_ckpt = join(opt.resume, 'checkpoint.pth.tar')
=======
    resume_ckpt = opt.resume
>>>>>>> Stashed changes

    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model = model.to(device)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_ckpt, checkpoint['epoch']))


    if cuda:
        model = nn.DataParallel(model)
        # model = model.to(device)

    data_path = './data/KITTIRot/'
    recall_seq = {"00":0, "02":0, "05":0, "06":0}
    for seq in list(recall_seq.keys()):
        print('===> Processing KITTI Seq. %s'%(seq))
        eval_set = dataset.KITTIDataset(data_path, seq)
        recalls = evaluate(eval_set, model)
        recall_seq[seq] = recalls[1]
    print("===> Recalls@1", recall_seq)