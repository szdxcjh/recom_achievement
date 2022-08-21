# -*- coding: utf-8 -*-
from nis import cat
from pkgutil import extend_path
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=10, dropout=[0.5, 0.5], 
                 use_cuda=True, verbose=False):
        """

        Inputs:
        - feature_size: 一个整数列表，给出每个字段特征大小(数据数目)
        - embedding_size: 嵌入向量维度
        - hidden_dims: 一个整数列表，给出隐藏层的维度
        - num_classes: 要预测的数目，例如1-5星
        - batch_size: 每次交互实例大小
        - use_cuda: 是否用cuda
        - verbose: 是否打印
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.float

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        """
        数据集中，连续数据过线性层/全连接层，离散数据过embedding层
        """
        first_dense_embedding = nn.ModuleList(
            [nn.Linear(size,self.embedding_size) for size in self.feature_sizes[:13]]
        )
        first_sparse_embedding = nn.ModuleList(
            [nn.Embedding(size,self.embedding_size) for size in self.feature_sizes[13:40]]
        )
        self.first_embedding = first_dense_embedding.extend(first_sparse_embedding)
        
        second_dense_embedding = nn.ModuleList(
            [nn.Linear(size,self.embedding_size) for size in self.feature_sizes[:13]]
        )
        second_sparse_embedding = nn.ModuleList(
            [nn.Embedding(size,self.embedding_size) for size in self.feature_sizes[13:40]]
        )
        self.second_embedding = second_dense_embedding.extend(second_sparse_embedding)
        
        """
        深度部分
        """
        
        all_dims = [self.field_size*self.embedding_size]+self.hidden_dims+[self.num_classes]
        for i in range(1,len(self.hidden_dims) + 1):
            setattr(self,'linear'+str(i),nn.Linear(all_dims[i-1],all_dims[i]))
            setattr(self,'bn'+str(i),nn.BatchNorm1d(all_dims[i]))
            setattr(self,'drop'+str(i),nn.Dropout(dropout[i-1]))
        
    def forward(self,Xi,Xv):
        """
        Args:
            Xi: 输入位置索引：(N, field_size, 1)
            Xv: 输入的值：(N, field_size, 1)
        """
        #拿个系数
        fm_first_emb = []
        for i,emb in enumerate(self.first_embedding):
            if i<13:
                Xi_ = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_first_emb.append((emb(Xi_).t() * Xv[:, i]).t())
            else:
                Xi_ = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_first_emb.append((torch.sum(emb(Xi_),1).t() * Xv[:, i]).t())
        #特征集合
        fm_first_emb_all = torch.cat(fm_first_emb, 1)
        fm_second_emb = []
        for i,emb in enumerate(self.second_embedding):
            if i<13:
                Xi_ = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_second_emb.append((emb(Xi_).t() * Xv[:, i]).t())
            else:
                Xi_ = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_second_emb.append((torch.sum(emb(Xi_),1).t() * Xv[:, i]).t())
        #xy=0.5*[(x+y)^(2)-(x^2+y^2)]
        fm_second_emb_all = sum(fm_second_emb)
        #(x+y)^(2)
        fm_sum_second_order_emb_square = fm_second_emb_all*fm_second_emb_all
        #x^2+y^2
        fm_sum_second_order_emb = [item*item for item in fm_second_emb]
        fm_second_order_emb_square_sum = sum(fm_sum_second_order_emb)
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        
        """deep"""
        deep_emb = torch.cat(fm_second_emb,1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self,'linear'+str(i))(deep_out)
            deep_out = getattr(self,'bn'+str(i))(deep_out)
            if(i != len(self.hidden_dims)):
                deep_out = getattr(self,'drop'+str(i))(deep_out)
                
        #三部分求和
        bias = torch.nn.Parameter(torch.randn(Xi.size(0)))
        total_sum = torch.sum(fm_first_emb_all, 1) + \
                    torch.sum(fm_second_order, 1) + \
                    torch.sum(deep_out, 1) + bias
                    
        return total_sum
    def fit(self, loader_train, loader_val, optimizer, epochs=1, verbose=False, print_every=5):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: 优化器 e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: .
        - verbose: 是否输出
        - print_every: 每次迭代后打印
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits

        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=self.dtype)
                
                total = model(xi, xv)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch %d Iteration %d, loss = %.4f' % (epoch, t, loss.item()))
                    self.check_accuracy(loader_val, model)
    
    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                total = model(xi, xv)
                preds = (F.sigmoid(total) > 0.5).to(dtype=self.dtype)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))