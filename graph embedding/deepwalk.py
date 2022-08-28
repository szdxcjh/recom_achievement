#deepwalk = randomwalk + skipgram
import networkx as nx
import random
import gensim
#构造有向图
def get_graph(nodes,edges):
    Graph = nx.DiGraph()
    Graph.add_nodes_from(nodes)
    Graph.add_edges_from(edges)
    return Graph

#随机游走
def random_walk(Graph,walk_nums:int=2,walk_length:int=3):
    nodes = list(Graph.nodes())
    res_list = []
    for walknum in range(walk_nums):
        random.shuffle(nodes)
        for node in nodes:
            node_list = []
            now_node = node
            node_list.append(now_node)
            for _ in range(walk_length-1):
                neighbors = list(Graph.neighbors(now_node))
                if(len(neighbors)>0):
                    now_node = random.choice(neighbors)
                    node_list.append(now_node)
                else:
                    break
            res_list.append(node_list)
    return res_list

if __name__ == "__main__":
    nodes = ['0', '1', '2', '3', '4', '5']
    edges = [('0', '0'), ('0', '1'), ('0', '5'),('3','1'),('1', '2'),('1', '4'),('2', '1'),('2', '4')]
    Graph = get_graph(nodes,edges)
    walk_list = random_walk(Graph,3,3)
    #sg=1  skip—gram  hs=1 层级softmax将会被使用,hs=0且negative不为0，则负采样将会被选择使用
    w2v = gensim.models.Word2Vec(walk_list,sg=1,hs=1,negative=0)
    print(w2v.wv['0'])