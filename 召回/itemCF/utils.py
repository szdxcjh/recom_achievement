# 根据点击时间获取用户的点击物品序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    """
    rank:用户购买物品的顺序排名
    vid_encode:物品id
    did_encode:用户id
    """
    click_df = click_df.sort_values('rank')
    
    def make_item_time_pair(df):
        return list(zip(df['vid_encode'], df['rank']))
    
    user_item_time_df = click_df.groupby('did_encode')\
        ['vid_encode', 'rank'].apply(lambda x: make_item_time_pair(x))\
        .reset_index().rename(columns={0: 'item_time_list'})
        
    user_item_time_dict = dict(zip(user_item_time_df['did_encode'],\
                               user_item_time_df['item_time_list']))
    
    return user_item_time_dict

#获取热门物品
def get_item_topk_click(click_df, k):
    topk_click = click_df['vid_encode'].value_counts().index[:k]
    return topk_click