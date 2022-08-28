#Line for first/second order achievement
#
from turtle import forward
import networkx as nx
import random
import torch.nn as nn
import torch
import torch.optim as optim

#模型
class Line(nn.Module):
    def __init__(self,graph,node_nums,embed_size):
        super().__init__()
        self.graph = graph
        self.embed_size = embed_size
        self.first_order_embedding = nn.Embedding(node_nums,embed_size)
        self.second_order_embedding = nn.Embedding(node_nums,embed_size)
    
    def forward(self,nodes):
        #字符串节点需要做编码
        #nodes = torch.tensor(list(self.graph.nodes))
        nodes = torch.tensor(list(map(int,nodes)))
        first_order_emb = self.first_order_embedding(nodes)
        first_order_combine_p = torch.exp(-torch.mm(first_order_emb,first_order_emb.T))
        first_order_combine_p = 1/(1 + first_order_combine_p)
        return first_order_combine_p




#构造无向图
def get_graph(nodes,edges):
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    Graph.add_weighted_edges_from(edges)
    return Graph

#定义经验分布
def frist_emp_distri(graph,edges):
    node_nums = graph.number_of_nodes()
    #nodes = torch.tensor(list(graph.nodes))
    emp_distri = torch.zeros((node_nums,node_nums))
    for edge in edges:
        node1 = int(edge[0])
        node2 = int(edge[1])
        emp_distri[node1][node2] = emp_distri[node2][node1] = edge[2]
    #emp_distri = emp_distri/torch.sum(emp_distri)
    return emp_distri
    
#KL散度忽略常数项
class first_order_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,first_order_combine_p,emp_distri):
        loss = - torch.sum(emp_distri * torch.log(first_order_combine_p))
        return loss
    

if __name__ == "__main__":
    nodes = ['0', '1', '2', '3', '4', '5']
    edges = [('0', '0',2), ('0', '1',3), ('0', '5',7),('3','1',5),('1', '2',4),
             ('1', '4',8),('2', '4',1)]
    Graph = get_graph(nodes,edges)
    node_nums = Graph.number_of_nodes()
    emp_distri = frist_emp_distri(Graph,edges)
    model = Line(Graph,node_nums,embed_size=8)
    epoches = 3
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    loss_func = first_order_loss()
    for epoch in range(epoches):
        optimizer.zero_grad()
        first_order_combine_p = model(nodes)
        loss = loss_func(first_order_combine_p,emp_distri)
        loss.backward()
        optimizer.step()
        print("loss:{}".format(loss))

