import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

# 900000 items for training, 10000 items for valid, of all 1000000 items
Num_train = 800

# load data
train_data = CriteoDataset('./data', train=True)
loader_train = DataLoader(train_data, batch_size=50,
                          sampler=sampler.SubsetRandomSampler(range(Num_train)))
val_data = CriteoDataset('./data', train=True)
loader_val = DataLoader(val_data, batch_size=50,
                        sampler=sampler.SubsetRandomSampler(range(Num_train, 899)))

feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

model = DeepFM(feature_sizes, use_cuda=False)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
model.fit(loader_train, loader_val, optimizer, epochs=100, verbose=True)

# test


#xi[:,36,:]
#d = defaultdict(int)
#from collections import defaultdict
#for ii in range(13, 39):
#    for t, (xi, xv, y) in enumerate(loader_train):
#        d = defaultdict(int)
#        for k in xi[:, ii, :]:
#            d[int(k)] += 1
#    print(d)
