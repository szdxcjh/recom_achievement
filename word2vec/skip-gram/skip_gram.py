#该部分只是对word2vec模型的简单实现，不涉及哈夫曼树编码等操作

import torch.nn as nn
import torch


class SkipGram(nn.Module):
    def __init__(self,vocabs,size):
        super().__init__()
        self.vocabs = torch.LongTensor(vocabs)
        self.word_embs = nn.Embedding(len(vocabs),size)
        #背景词向量
        self.neigh_embs = nn.Embedding(len(vocabs),size)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.word_embs(x)
        neigh = self.neigh_embs(self.vocabs)
        y = torch.mm(x,neigh.T)
        y = self.softmax(y)
        return y

def getvocabsOnltIndex(seqs):
    #所有词的集合
    vocabs = set()
    for seq in seqs:
        vocabs |= set(seq)
    vocabs = list(vocabs)
    return vocabs

def train_word2vec(seqs,window_size=1):
    vocabs = getvocabsOnltIndex(seqs)
    net = SkipGram(vocabs,16)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    net.train()
    for seq in seqs:
        for word_num in range(window_size,len(seq)-window_size):
            optimizer.zero_grad()
            window = seq[word_num-window_size:word_num+window_size+1]
            x = torch.LongTensor([window[window_size] for _ in range(window_size*2)])
            y_pre = net(x)
            #剔除中心词，判断左右词
            window.pop(window_size)
            y = torch.LongTensor(window)
            loss = loss_func(y_pre,y)
            loss.backward()
            optimizer.step()
            print("loss:{}".format(loss))


if __name__ == '__main__':
    seqs = [[0,3,7,6],
            [1,5,7,4],
            [3,2,6,8],
            [5,3,6,9]]
    train_word2vec(seqs)