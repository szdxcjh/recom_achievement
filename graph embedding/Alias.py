#Alias sampling
#不同的桶分为不同的颜色，让所有桶都至多有两种颜色，o(1)的调用，o(k)的构建

import random
import math
def alisa_struct(prob_list):
    K = len(prob_list)
    new_prob_list = [item*K for item in prob_list]
    ori_prob = [0 for i in range(K)]
    add_id = [0 for i in range(K)]
    smaller = []
    larger = []
    for i in range(K):
        ori_prob[i] = new_prob_list[i]
        if(new_prob_list[i] > 1):
            larger.append(i)
        if(new_prob_list[i] < 1):
            smaller.append(i)
    while(len(smaller) > 0):
        small_id = smaller.pop()
        large_id = larger.pop()
        add_id[small_id] = large_id
        ori_prob[large_id] -= 1-ori_prob[small_id]
        if(ori_prob[large_id] < 1):
            smaller.append(large_id)
        if(ori_prob[large_id] > 1):
            larger.append(large_id)
    return ori_prob,add_id
if __name__ == "__main__":
    prob_list = [0.2,0.1,0.3,0.4]
    ori_prob,add_id = alisa_struct(prob_list)
    bucket = math.floor(random.uniform(0,1) * len(prob_list))
    if(random.uniform(0,1) <= ori_prob[bucket]):
        print(bucket)
    else:
        print(add_id[bucket])
