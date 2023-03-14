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
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='BEVPlace')
parser.add_argument('--test_batch_size', type=int, default=128, help='Batch size for testing')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=40, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='checkpoints', help='Path to load checkpoint from, for resuming training or testing.')


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
                batch_feature = bevplace(input)
                global_features.append(batch_feature.detach().cpu().numpy())

                t.update(1)

    global_features = np.vstack(global_features)

    query_feat = global_features[eval_set.num_db:].astype('float32')
    db_feat = global_features[:eval_set.num_db].astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(query_feat.shape[1])
    faiss_index.add(db_feat)

    print('====> Calculating recall @ N')
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

    recalls = {} 
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return recalls

import dataset
if __name__ == "__main__":
    opt = parser.parse_args()

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")


    print('===> Loading dataset(s)')
    data_path = './data/KITTI05/'
    seq = '05'
    eval_set = dataset.KITTIDataset(data_path, seq)
     
    print('===> Building model')
    bevplace = BEVPlace()
    resume_ckpt = join(opt.resume, 'checkpoint.pth.tar')

    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    bevplace.load_state_dict(checkpoint['state_dict'])
    bevplace = bevplace.to(device)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_ckpt, checkpoint['epoch']))


    bevplace = nn.DataParallel(bevplace)
    model = bevplace.to(device)

    recalls = evaluate(eval_set, model)
    