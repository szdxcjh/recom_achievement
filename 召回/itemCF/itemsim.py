import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict 
from tqdm import tqdm
import math
from utils import *
import pickle
import numpy as np

def get_item_sim(df,user_item_time_dict,save=True):

    i2i_smi = dict()
    item_cnt = defaultdict(int)

    for user_id,item_time_list in tqdm(user_item_time_dict.items()):
        for (i,rank1) in item_time_list:
            item_cnt[i] += 1
            i2i_smi.setdefault(i,{})
            for (j,rank2) in item_time_list:
                if(i == j):
                    continue
                i2i_smi[i].setdefault(j,0)
                i2i_smi[i][j] += 1
    i2i_smi_ = i2i_smi.copy()
    #归一化
    for i,sim_dict in i2i_smi.items():
        for j,sim in sim_dict.items():
            i2i_smi[i][j] = i2i_smi[i][j]/math.sqrt(item_cnt[i]*item_cnt[j])
    #保存
    if(save):
        pickle.dump(i2i_smi, open('itemcf_i2i_sim.pkl', 'wb'))
    return i2i_smi
