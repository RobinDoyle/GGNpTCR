import numpy as np
import pandas as pd
import os,sys
import pickle
import timeit
from collections import defaultdict

from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

aaindex = np.array([
    #  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V    X
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

after_pca = np.loadtxt('./dataset/after_pca.txt')
aaindex = np.hstack((aaindex, after_pca))

def Protein2Sequence(sequence, ngram=1):
    # convert sequence to CNN input
    sequence = sequence.upper()
    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(20)
        else:
            output.append(aa_list.index(word))
    if ngram == 3:
        output = [-1]+output+[-1] # pad
    return np.array(output, np.int32)

def pack1D(in_list, N, length, dim, dtype = torch.float):
    output = torch.zeros((N, length, dim), dtype = dtype)
    i = 0
    for out in in_list:
        a_len = out.shape[0]
        output[i, :a_len, :] = out
        i += 1
    return output

def pack(seq1, seq2, label, score, device):
    seq1_len = 0
    seq2_len = 0
    N = len(seq1)
    dim = seq1[0].shape[1]
    seq1_num = []
    seq2_num = []
    for seq in seq1:
        seq1_num.append(seq.shape[0])
        if seq.shape[0] >= seq1_len:
            seq1_len = seq.shape[0]
    for seq in seq2:
        seq2_num.append(seq.shape[0])
        if seq.shape[0] >= seq2_len:
            seq2_len = seq.shape[0]
    out_seq1 = pack1D(seq1, N, seq1_len, dim).to(device)
    out_seq2 = pack1D(seq2, N, seq2_len, dim).to(device)
    out_label = torch.IntTensor(label).to(device)
    out_score = torch.FloatTensor(score).to(device)
    return out_seq1, out_seq2, out_label, out_score, seq1_num, seq2_num

def mask_make(num, seq, device):
    l = seq.shape[1]
    N = len(num)
    mask = torch.zeros((N, l))
    for i in range(N):
        mask[i, :num[i]] = 1
    mask = mask.to(device)
    return mask

class Trainer(object):
    def __init__(self, model, lr, weight_decay, bsz, device):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.bsz = bsz
        self.device = device
    def train(self, dataset):#seq1, seq2, label, score):
        #criterion = nn.CrossEntropyLoss()
        N = len(dataset)
        i = 0
        loss_total = 0
        self.optimizer.zero_grad()
        seqs1, seqs2, labels, scores = [], [], [], []
        for data in dataset:
            i = i + 1
            seq1, seq2, label, score = data
            seqs1.append(seq1)
            seqs2.append(seq2)
            labels.append(label)
            scores.append(score)
            if i % self.bsz == 0 or i == N:
                in_seq1, in_seq2, in_label, in_score, num_seq1, num_seq2 = pack(seqs1, seqs2, labels, scores, self.device)
                bsz = in_seq1.size()[0]
                mask1 = mask_make(num_seq1, in_seq1, self.device)
                mask2 = mask_make(num_seq2, in_seq2, self.device)
                out_score = self.model(in_seq1, in_seq2, mask1, mask2, bsz).squeeze(1)
                loss = F.binary_cross_entropy(out_score, in_score)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_total = loss_total + bsz * loss
                seqs1, seqs2, labels, scores = [], [], [], []
        return loss_total/N

class Tester(object):
    def __init__(self, model, bsz, device):
        self.model = model
        self.bsz = bsz
        self.device = device
    def test(self, dataset):#seq1, seq2, label, score):
        N = len(dataset)
        i = 0
        seqs1, seqs2, labels, scores = [], [], [], []
        T, Y, S = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        with torch.no_grad():
            for data in dataset:
                i = i + 1
                seq1, seq2, label, score = data
                seqs1.append(seq1)
                seqs2.append(seq2)
                labels.append(label)
                scores.append(score)
                if (i+1) % self.bsz == 0 or (i+1) == N:
                    in_seq1, in_seq2, in_label, in_score, num_seq1, num_seq2 = pack(seqs1, seqs2, labels, scores, self.device)
                    bsz = in_seq1.size()[0]
                    mask1 = mask_make(num_seq1, in_seq1, self.device)
                    mask2 = mask_make(num_seq2, in_seq2, self.device)
                    out_score = self.model(in_seq1, in_seq2, mask1, mask2, bsz)
                    T = torch.cat((T, in_label), 0)
                    S = torch.cat((S, out_score), 0)
                    seqs1, seqs2, labels, scores = [], [], [], []
        T = T.to('cpu').data.numpy()
        S = S.to('cpu').data.numpy()
        Y = np.round(S)
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        
        return AUC, PRC, precision, recall
    
    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def val(data, n_val, i) :
    l=int(len(data)/n_val)
    if i==n_val-1:
        val=data[(l*i):]
        train=data[:(l*i)]
    elif i==0:
        val=data[:l]
        train=data[l:]
    else :
        val=data[(l*i):(l*(i+1))]
        train=data[:(l*i)]+data[l*(i+1):]
    return val, train

ori = pd.read_csv('./dataset/majority_training_dataset.csv')
ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))

word_dict = defaultdict(lambda: len(word_dict))
input1 = []
input2 = []
label = []
score = []

for i in range(ori.shape[0]):
    temp1 = ori.peptide[i]
    temp2 = ori.binding_TCR[i]
    immun = ori.label[i]
    input1.append(torch.from_numpy(aaindex[Protein2Sequence(temp1),]))
    input2.append(torch.from_numpy(aaindex[Protein2Sequence(temp2),]))
    label.append(immun)
    score.append(immun)

label = torch.tensor(label)
score = torch.Tensor(score)

n_epoch = 50
batch_size = 128
x_layer = 3
y_layer = 3
out_layer = 3
in_dim = 32
dim = 128
lr = 0.001
lr_decay = 0.5
weight_decay = 1e-05
n_val = 10

dataset = list(zip(input1, input2, label, score))
dataset = shuffle_dataset(dataset, 0)

ori = pd.read_csv('./dataset/majority_testing_dataset.csv')
ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))

word_dict = defaultdict(lambda: len(word_dict))
input1 = []
input2 = []
label = []
score = []

for i in range(ori.shape[0]):
    temp1 = ori.peptide[i]
    temp2 = ori.binding_TCR[i]
    immun = ori.label[i]
    input1.append(torch.from_numpy(aaindex[Protein2Sequence(temp1),]))
    input2.append(torch.from_numpy(aaindex[Protein2Sequence(temp2),]))
    label.append(immun)
    score.append(immun)

label_ = torch.tensor(label)
score = torch.Tensor(score)

dataset_test = list(zip(input1, input2, label_, score))

file_result = "./output/test.txt"

result = ('Val\tEpoch\tTime\tLoss_train\tAUC_dev\tPRC_dev\tAUC_test\tPRC_test')
with open(file_result, 'w') as f:
    f.write(result + '\n')

for k in range(n_val):
    file_model = "./output/test" + str(k)
    dataset_dev, dataset_train = val(dataset, n_val, k)
    torch.manual_seed(k)
    model = Net(geom_graph_attn, attn_graph_attn, in_dim = in_dim, dim = dim, x_layer = x_layer, y_layer = y_layer, out_layer = out_layer, device = device).to(device)
    trainer = Trainer(model, lr, weight_decay, batch_size, device)
    tester = Tester(model, batch_size, device)
    max_auc = 0
    temp = 0
    start = timeit.default_timer()
    for i in range(n_epoch):
        loss_train = trainer.train(dataset_train)#(input1, input2, label, score)
        AUC, PRC, _, _ = tester.test(dataset_dev)#(input1, input2, label, score)
        AUC_test, PRC_test, _, _ = tester.test(dataset_test)
        AUC_test1, PRC_test1, _, _ = tester.test(dataset_test1)
        end = timeit.default_timer()
        time = end - start
        result = [k, i+1, time, loss_train.to('cpu').data.numpy(), AUC, PRC, AUC_test, PRC_test, AUC_test1, PRC_test1]
        print('\t'.join(map(str, result)))
        with open(file_result, 'a') as f:
            f.write('\t'.join(map(str, result)) + '\n')
        if temp > AUC:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        if AUC_test>max_auc:
            max_auc = AUC_test
            tester.save_model(model, file_model)
        if np.abs(temp - AUC) < 1e-4:
            break
        temp = AUC

T_s, S_s = torch.tensor([]).to(device), torch.tensor([]).to(device)

for k in range(n_val):
    file_model = "./output/test" + str(k)
    model = Net(geom_graph_attn, attn_graph_attn, in_dim = in_dim, dim = dim, x_layer = x_layer, y_layer = y_layer, out_layer = out_layer, device = device).to(device)
    model.load_state_dict(torch.load(file_model))
    i = 0
    N = len(dataset_test)
    seqs1, seqs2, labels, scores = [], [], [], []
    T, S = torch.tensor([]).to(device), torch.tensor([]).to(device)
    with torch.no_grad():
        for data in dataset_test:
            i = i + 1
            seq1, seq2, label, score = data
            seqs1.append(seq1)
            seqs2.append(seq2)
            labels.append(label)
            scores.append(score)
            if (i+1) % batch_size == 0 or i == N:
                in_seq1, in_seq2, in_label, in_score, num_seq1, num_seq2 = pack(seqs1, seqs2, labels, scores, device)
                bsz = in_seq1.size()[0]
                mask1 = mask_make(num_seq1, in_seq1, device)
                mask2 = mask_make(num_seq2, in_seq2, device)
                out_score = model(in_seq1, in_seq2, mask1, mask2, bsz)
                T = torch.cat((T, in_label), 0)
                S = torch.cat((S, out_score), 0)
                seqs1, seqs2, labels, scores = [], [], [], []
    T_ = T.to('cpu').data.numpy()
    S_ = S.to('cpu').data.numpy()
    AUC = roc_auc_score(T_, S_)
    tpr, fpr, _ = precision_recall_curve(T_, S_)
    PRC = auc(fpr, tpr)
    result = [AUC, PRC]
    print('\t'.join(map(str, result)))
    T = torch.unsqueeze(T, dim=0)
    S = torch.unsqueeze(S, dim=0)
    T_s = torch.cat((T_s, T), dim=0)
    S_s = torch.cat((S_s, S), dim=0)

T = torch.mean(T_s, dim = 0)
S = torch.mean(S_s, dim = 0)
T = T.to('cpu').data.numpy()
S = S.to('cpu').data.numpy()
AUC = roc_auc_score(T, S)
tpr, fpr, _ = precision_recall_curve(T, S)
PRC = auc(fpr, tpr)
result = [AUC, PRC]
print('\t'.join(map(str, result)))
with open(file_result, 'a') as f:
    f.write('\t'.join(map(str, result)) + '\n')
