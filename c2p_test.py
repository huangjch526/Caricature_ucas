'''
code version 1.0 by hjc (from nju to ucas)
'''

from __future__ import print_function
from torchvision import transforms
import argparse
import bisect
import datetime
import os
import pickle
import random
import sys
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from net_sphere import sphere20a

torch.backends.cudnn.bencmark = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch sphereface wc')
    parser.add_argument('--dataset_path', default='./test/', type=str)
    parser.add_argument('--class_num', default=123, type=int)
    # parser.add_argument('--model', '-m', default='../../support_material/sphere20a.pth', type=str)
    parser.add_argument('--model', '-m', default='outputs\init_original_dataset\checkpoints/00012000.pth', type=str)
    # parser.add_argument('--model', '-m', default='outputs/init_original_dataset_SE3out/checkpoints/00005000.pth', type=str)

    parser.add_argument('--ipython', action='store_true')
    parser.add_argument('--myGpu', default='0', help='GPU Number')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.myGpu

    if args.ipython:
        from IPython import embed; embed()
        exit(0)

    # 加载网络
    net = sphere20a(classnum=args.class_num)
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    net.feature = True

    transform = transforms.Compose([
        transforms.Resize((112, 96)),
    ])

    # 加载Probe图片数据
    with open('FR_Probe_C2P.txt') as f:
        probe_lines = f.readlines()
    # 对每一行进行处理
    probes=[]
    for line in probe_lines:
        l = line.strip()
        probes.append(l)

    # 加载Gallery
    with open('FR_Gallery_C2P.txt') as f:
        gallery_lines = f.readlines()
        # 对每一行进行处理
    gallerys = []
    for line in gallery_lines:
        l = line.strip()
        gallerys.append(l)

    predicts=[]
    gallerys_feature=[]
    for k in range(len(gallerys)):
        this_gallery_path = args.dataset_path + gallerys[k] + '.jpg'

        gallery_img = np.array(transform(Image.open(this_gallery_path).convert('RGB')))

        imglist = [gallery_img, cv2.flip(gallery_img, 1)]  # cv2.flip(img1, 1):112*96*3
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            imglist[i] = (imglist[i] - 127.5) / 128.

        img = np.vstack(imglist)  # 2*3*112*96  (垂直将imglist中的2组数据合起来)
        with torch.no_grad():
            img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
            output = net(img)  # 2*512
        f = output.data
        f1 = f[0]  # img1 output
        gallerys_feature.append(f1)

    for i in range(len(probes)):
        print(str(i) + " start")
        this_probe_path = args.dataset_path+probes[i]+'.jpg'

        probe_img = np.array(transform(Image.open(this_probe_path).convert('RGB')))

        imglist = [probe_img, cv2.flip(probe_img, 1)]  # cv2.flip(img1, 1):112*96*3
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            imglist[i] = (imglist[i] - 127.5) / 128.

        img = np.vstack(imglist)  # 2*3*112*96  (垂直将imglist中的2组数据合起来)
        with torch.no_grad():
            img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
            output = net(img)  # 2*512
        f = output.data
        f1 = f[0]  # img1 output

        this_probe_scores=[]
        for j in range(len(gallerys)):

            f2=gallerys_feature[j]
            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            this_probe_scores.append(cosdistance)

        this_probe_scores=np.array(this_probe_scores)

        max_index=this_probe_scores.argmax()

        most_pos_gallery=gallerys[max_index]

        predicts.append(most_pos_gallery)


    predicts=np.array(predicts).reshape(len(probes),1)
    predicts = pd.DataFrame(predicts)
    predicts.to_csv('result.csv', header=0, index=0)





