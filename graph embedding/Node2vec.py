#Node2vec = 概率采样 + deepwalk
#当第n个节点的邻居是n-1个节点的邻居时概率为1，否则为1/q,返回概率为1/p
#先采样顶点，后用deepwalk
#邻近点和中心点用的一套embedding向量
#alias采样用numpy代替
import networkx as nx
import random
import gensim
import numpy as np
#构造有向图
def get_graph(nodes,edges):
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    Graph.add_edges_from(edges)
    return Graph

#Node2vec
def Node2vec(Graph,p,q,walk_nums=2,walk_length=4):
    nodes = list(Graph.nodes())
    res_list = []
    for walknum in range(walk_nums):
        random.shuffle(nodes)
        for node in nodes:
            node_list = []
            #先加入两个节点
            last_node = node
            node_list.append(last_node)
            last_neighbors = list(Graph.neighbors(last_node))
            if(len(last_neighbors)>0):
                now_node = random.choice(last_neighbors)
                node_list.append(now_node)
            else:
                res_list.append(node_list)
                continue
            for _ in range(2,walk_length):
                neighbors = list(Graph.neighbors(now_node))
                if(len(neighbors)>0):
                    choice_prob = []
                    for neighbor in neighbors:
                        if(neighbor in last_neighbors):
                            choice_prob.append(1)
                        elif(neighbor == last_node):
                            choice_prob.append(1/p)
                        else:
                            choice_prob.append(1/q)
                    choice_prob = np.array(choice_prob)
                    choice_prob = choice_prob/sum(choice_prob)
                    next_id_list = np.random.multinomial(n=1,pvals=choice_prob,size=1)
                    next_id = list(np.nonzero(next_id_list)[1])[0]
                    last_node = now_node
                    last_neighbors = neighbors
                    now_node = neighbors[next_id]
                    node_list.append(now_node)
                else:
                    break
            res_list.append(node_list)
    return res_list

if __name__ == "__main__":
    nodes = ['0', '1', '2', '3', '4', '5', '6']
    edges = [('0', '1'), ('0', '5'),('3','1'),('4','6'),('2','6'),
             ('1', '2'),('1', '4'),('2', '1'),('2', '4'),('5','2'),('3','6')]
    Graph = get_graph(nodes,edges)
    walk_list = Node2vec(Graph,2,3,3,4)
    #sg=1  skip—gram  hs=1 层级softmax将会被使用,hs=0且negative不为0，则负采样将会被选择使用
    w2v = gensim.models.Word2Vec(walk_list,sg=1,hs=1,negative=0)
    print(w2v.wv['0'])