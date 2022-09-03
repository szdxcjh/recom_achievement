
from tqdm import tqdm
import sys
import pandas as pd
import pickle
import collections
import sys 
from utils import *
from itemsim import *
def item_cf(user_id,user_item_time_dict,i2i_sim,sim_item_topk,
            recall_item_num,item_topk_click):
    """
    基于文章协同过滤的召回
    :param user_id: 用户id
    :param user_item_time_dict: 字典, 根据点击时间获取用户的点击物品序列：
    {user1: [(item1, time1), (item2, time2)..]...}
    :param i2i_sim: 字典，文章相似性矩阵
    :param sim_item_topk: 整数， 选择与当前物品最相似的前k个物品
    :param recall_item_num: 整数， 最后的召回物品数量
    :param item_topk_click: 列表，点击次数最多的物品列表，用户召回补全
    
    return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """
    user_clicked_items = user_item_time_dict[user_id]
    #转为set
    user_clicked_items = {itemid for (itemid,rank) in user_clicked_items}

    item_smi_dict = {}
    for item_id in user_clicked_items:
        sim_items = sorted(i2i_sim[item_id].items(), key=lambda x:x[1],\
                           reverse=True)[:sim_item_topk]
        for (smi_item_id,smi_num) in sim_items:
            #用户点击过的不加入
            if(smi_item_id in user_clicked_items):
                continue
            item_smi_dict.setdefault(smi_item_id,0)
            item_smi_dict[smi_item_id] += smi_num
    
    #热门商品补全
    if len(item_smi_dict)<recall_item_num:
        for i,item_id in enumerate(item_topk_click):
            if(item_id in item_smi_dict.keys()):
                continue
            #递减相似度
            item_smi_dict[item_id] = -i
            if(len(item_smi_dict) == recall_item_num):
                break
    item_smi_res = sorted(item_smi_dict.items(),key=lambda x:x[1],reverse=True)[:recall_item_num]
    return item_smi_res

if __name__ == "__main__":
    all_click_df = pd.read_csv("./召回/itemCF/main_vv_seq_train_last4.csv")
    #这里只获取用户最后一个点击的序列，减少复杂度
    user_item_time_dict = get_user_item_time(all_click_df)

    #拿一个i2i_sim矩阵
    try:
        i2i_sim = pickle.load(open('itemcf_i2i_sim.pkl', 'rb'))
    except:
        i2i_sim = get_item_sim(all_click_df,user_item_time_dict,True)
    
    sim_item_topk = 50
    recall_item_num = 50
    #获取热门物品
    item_topk_click = get_item_topk_click(all_click_df, k=50)

    user_item_sim = dict()
    for user_id in tqdm(user_item_time_dict.keys()):
        user_item_sim[user_id] = item_cf(user_id,user_item_time_dict,i2i_sim,sim_item_topk,
            recall_item_num,item_topk_click)

    print("user{}推荐的物品为:{}".format(0,dict(user_item_sim[0]).keys()))
