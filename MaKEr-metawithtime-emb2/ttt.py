import pickle
import numpy as np
from utils import get_g, serialize
import torch
import lmdb
import dgl
from collections import defaultdict as ddict
from tqdm import tqdm
import random
from scipy import sparse
import multiprocessing as mp
import numpy as np
# from scipy.sparse import coo_matrix
# row  = np.array([0, 1, 1, 0])
# col  = np.array([0, 1, 3, 2])
# data = np.array([4, 5, 7, 9])
# a = coo_matrix((data, (row, col)))
# # array([[4, 0, 9, 0],
# #        [0, 7, 0, 0],
# #        [0, 0, 0, 0],
# #        [0, 0, 0, 5]])
# print(a)
# print(a.row)
# print(a.col)




# data = pickle.load(open('./data/fb_ext.pkl', 'rb'))
# a = data['test']['support']
# rel = dict()
# for i , tri in enumerate(a):
#     if (tri[0],tri[2])  in rel:
#         rel[(tri[0],tri[2])]+=1
#     else:
#         rel[(tri[0], tri[2])] = 1
# print(rel)


# for ent in range(0, 10):
#     print(range(0, 10))
#     print(1)
#     for time in range(0, 10):
#         print("time")
#         print(time)
#         print(2)
x_qian_y = torch.zeros((1, 10), dtype=torch.int)
x_hou_y = torch.zeros((1, 10), dtype=torch.int)
x_tong_y = torch.zeros((1, 10), dtype=torch.int)
rel_time = torch.ones(10,15)
for ent in range(0, 10):
    for time in range(0, 15):
        hou_zong = torch.zeros((1, 10), dtype=torch.float)
        qian_zong = torch.zeros((1, 10), dtype=torch.float)
        tong_zong = torch.zeros((1, 10), dtype=torch.float)
        if True:
            hou, tong, qian = torch.split(rel_time, [time, 1, 15 - 1 - time], 1)
            # print("hou = ", hou)
            print("torch.sum(hou, 1).unsqueeze(0)=", torch.sum(hou, 1).unsqueeze(0))
            print("===")
            print(hou_zong+ torch.sum(hou, 1).unsqueeze(0))
            hou_zong += torch.sum(hou, 1).unsqueeze(0)
            # print("hou_zong= ", hou_zong)
            qian_zong += torch.sum(qian, 1).unsqueeze(0)
            tong_zong += torch.sum(tong, 1).unsqueeze(0)
    x_qian_y = torch.cat((x_qian_y, qian_zong), 0)
    x_hou_y = torch.cat((x_hou_y, hou_zong), 0)
    x_tong_y = torch.cat((x_tong_y, tong_zong), 0)































