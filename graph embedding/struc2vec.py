"""
struc2vec:
1.获取节点，以及他们的邻居
"""
import networkx as nx
import random
import gensim
import collections
import numpy as np
#构造无向图
def get_graph(nodes,edges):
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    Graph.add_edges_from(edges)
    return Graph
#计算DTW距离
def DTW(listA,listB):
    max_a = max(listA)
    min_a = min(listA)
    max_b = max(listB)
    min_b = min(listB)
    return (max(max_a,max_b)/min(min_a,min_b))-1
#输入节点的列表，输出该列表的度列表
def get_degree_list(node_list,Graph):
    degree_list = np.zeros(len(node_list))
    i = 0
    for item in node_list:
        degree_list[i] = len(Graph.edges(item))
        i += 1
    return degree_list
#获得w矩阵
def get_distance(nodes,node_degree_list):
    layer = np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        for j in range(i,len(nodes)):
            node_i = node_degree_list[nodes[i]]
            node_j = node_degree_list[nodes[j]]
            layer[i][j] = layer[j][i] = DTW(node_i,node_j)
    return layer

#获取每一层节点后返回每一层的w矩阵
def node_diatance(nodes,Graph,k):
    now_neighbor_dict = collections.defaultdict(list)
    all_set_dict = collections.defaultdict(set)
    layer_matrix = collections.defaultdict()
    node_degree_list = collections.defaultdict()
    #temp:顶点集合
    for i in range(k):
        if(i == 0):
            for node in nodes:
                temp = np.array(list(node))
                now_neighbor_dict[node] = temp
                all_set_dict[node].add(node)
                node_degree_list[node] = get_degree_list(temp,Graph)
            layer_matrix[i] = get_distance(nodes,node_degree_list)

        else:
            for node in nodes:
                temp = []
                for neigh in now_neighbor_dict[node]:
                    temp += list(Graph.neighbors(neigh))
                temp = set(temp)
                temp -= all_set_dict[node]
                all_set_dict[node].update(temp)
                temp = np.array(list(temp))
                now_neighbor_dict[node] = temp
                node_degree_list[node] = get_degree_list(temp,Graph)
            layer_matrix[i] = get_distance(nodes,node_degree_list)+layer_matrix[i-1]
    return layer_matrix

#获取不同层的转移概率
def get_layer_weight(weight_matrix):
    layer_nodeweight = collections.defaultdict(list)
    p_nextlayer = collections.defaultdict(list)
    p_beforelayer = collections.defaultdict(list)
    for k in range(len(weight_matrix)):
        mean_weight = np.mean(weight_matrix[k])
        layer_nodeweight[k] = np.log(np.sum(weight_matrix[k]>mean_weight,axis=1)+np.e)
        p_nextlayer[k] = layer_nodeweight[k]/(layer_nodeweight[k]+1)
        p_beforelayer[k] = 1 - p_nextlayer[k]
    return p_nextlayer,p_beforelayer

#将w矩阵变为p矩阵
def w_to_p(weight_matrix):
    possi_matrix = collections.defaultdict()
    for k in range(len(weight_matrix)):
        Z = np.sum(weight_matrix[k],axis=1) - np.diagonal(weight_matrix[k])
        Z = Z.reshape((len(Z),-1))
        possi_matrix[k] = weight_matrix[k] / Z
    return possi_matrix

#节点采样
def get_sampling(p_nextlayer,possi_matrix,index,walk_length,start_layer,change_layer):
    if(change_layer):
        random_num = np.random.rand(1)
        if(start_layer < len(p_nextlayer)-1 and random_num <= p_nextlayer[start_layer][index]):
            start_layer += 1
        if(start_layer > 0 and random_num > p_nextlayer[start_layer][index]):
            start_layer -= 1
    sample_matrix = possi_matrix[start_layer].copy()
    row, col = np.diag_indices_from(sample_matrix)
    sample_matrix[row,col] = 0
    #随机游走
    walk_list = []
    walk_list.append(index)
    now_node = index
    for i in range(walk_length):
        next_id_list = np.random.multinomial(n=1,pvals=sample_matrix[now_node],size=1)
        next_id = list(np.nonzero(next_id_list)[1])[0]
        now_node = next_id
        walk_list.append(next_id)
    return walk_list
if __name__ == "__main__":
    nodes = ['0', '1', '2', '3', '4', '5','6']
    edges = [('0', '1'), ('0', '5'),('3','1'),('1', '2'),('1', '4'),
             ('2', '1'),('2', '4'),('5','2'),('6','3'),('6','4'),('6','2')]
    #索引和节点的关系映射
    index_node = dict()
    for index,node in enumerate(nodes):
        index_node[index] = node
    Graph = get_graph(nodes,edges)
    #层数
    k = 2
    layer_matrix = node_diatance(nodes,Graph,k)
    #wk
    weight_matrix = collections.defaultdict()
    for key,value in layer_matrix.items():
        weight_matrix[key] = np.exp(-value)
    p_nextlayer,p_beforelayer = get_layer_weight(weight_matrix)
    possi_matrix = w_to_p(weight_matrix)
    walk_num = 2
    walk_length = 3
    res = []
    for _ in range(walk_num):
        for index in range(len(nodes)):
            walk_list = get_sampling(p_nextlayer,
                                    possi_matrix,
                                    index,
                                    walk_length,
                                    np.random.randint(2),
                                    change_layer=True)
            res.append(walk_list)
    for i in range(len(res)):
        for j in range(len(res[0])):
            res[i][j] = index_node[res[i][j]]
    #sg=1  skip—gram  hs=1 层级softmax将会被使用,hs=0且negative不为0，则负采样将会被选择使用
    w2v = gensim.models.Word2Vec(res,sg=1,hs=1,negative=0)
    print(w2v.wv['2'])