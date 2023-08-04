import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

class geom_graph_attn(nn.Module):
    def __init__(self, in_dim=20, dim=128, n_heads=8, kernel_size=3, kernel_size_2d=(3,3), dilation=(1,1), stride=1, drop_rate=0.1, layer=3, device=device):
        super(geom_graph_attn, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.layer = layer
        self.scale = torch.sqrt(torch.FloatTensor([dim // n_heads])).to(device)
        self.do = nn.Dropout(drop_rate)
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layer)])

        self.conv1 = nn.Conv1d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=False, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=False, padding=kernel_size//2)
        
        self.conv1d = nn.ModuleList([nn.Conv1d(dim, dim, kernel_size=kernel_size,
                               stride=stride, bias=False, padding=kernel_size//2) for _ in range(layer)])
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(layer)])
        
        self.conv2d = nn.ModuleList([nn.Conv2d(2*dim, 2*dim, kernel_size=kernel_size_2d, dilation=dilation, stride=(stride, 1),
                               padding=(((kernel_size_2d[0] - 1) * dilation[0])//2, ((kernel_size_2d[0] - 1) * dilation[0])//2), bias=False) for _ in range(layer)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(2*dim) for _ in range(layer)])

        self.w_q = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.w_k = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.w_v = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.w_d = nn.ModuleList([nn.Linear(2*dim, n_heads) for _ in range(layer)])
        self.w_l = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])

        self.fc = nn.Linear(dim, dim)

    def softmax(self, a, mask, dim=-1):
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax

    def seq2pair(self, x):
        x = torch.transpose(x, 1, 2)

        # Vertical expansion, convert to bxLxLxn where out_tensor[b][i][j] = in_matrix[b][i]
        vert_expansion = x.clone()
        vert_expansion.unsqueeze_(2)
        vert_expansion = vert_expansion.expand(vert_expansion.shape[0], vert_expansion.shape[1],
                                               vert_expansion.shape[1], vert_expansion.shape[3])

        # For every b, i, j pair, append in_matrix[b][j] to out_tensor[b][i][j]
        x_shape = x.shape
        pair = x.clone()
        pair.unsqueeze_(1)
        pair = pair.expand(pair.shape[0], x_shape[1], pair.shape[2], pair.shape[3])
        out_tensor = torch.cat([vert_expansion, pair], dim=3)

        # Switch shape from [batch, timestep/length_i, timestep/length_j, filter/channel]
        #                to [batch, filter/channel, timestep/length_i, timestep/length_j]
        out_tensor = torch.einsum('bijc -> bcij', out_tensor)

        return out_tensor

    def forward(self, seq, mask=None, bsz=16):
        seq = seq.transpose(1, 2)
        x = self.conv1(seq).transpose(1, 2)
        c = self.conv2(seq)
        c2 = self.seq2pair(c)

        for i in range(self.layer):
            c = F.relu(self.bn1[i](self.conv1d[i](c)))
            x = self.norm[i](x + c.transpose(1, 2))
            c2 = c2 + self.seq2pair(c)
            c2 = F.relu(self.bn2[i](self.conv2d[i](c2)))
            d = c2 + c2.transpose(2, 3)
            d = torch.sigmoid(self.w_d[i](d.transpose(1, 3)))
            
            q = F.relu(self.w_q[i](x)).view(bsz, -1, self.n_heads, self.dim//self.n_heads).permute(0,2,1,3)
            k = F.relu(self.w_k[i](x)).view(bsz, -1, self.n_heads, self.dim//self.n_heads).permute(0,2,1,3)
            v = F.relu(self.w_v[i](x)).view(bsz, -1, self.n_heads, self.dim//self.n_heads).permute(0,2,1,3)

            attn = torch.matmul(q,k.permute(0,1,3,2))/self.scale
            attn = self.softmax(attn, mask.view(bsz, 1, 1, -1))
            attn = attn*d.transpose(1,3)

            update = torch.matmul(attn, v)
            update = update.permute(0,2,1,3).contiguous().view(bsz, -1, self.n_heads*(self.dim//self.n_heads))
            x = x + self.do(update)
            x = F.relu(self.w_l[i](x))

        x = F.relu(self.fc(x))

        return x

class attn_graph_attn(nn.Module):
    def __init__(self, dim=128, n_heads=8, drop_rate=0.1, layer=3, device=device):
        super(attn_graph_attn, self).__init__()
        self.n_heads = n_heads
        self.layer = layer
        self.scale = torch.sqrt(torch.FloatTensor([dim])).to(device)
        self.do = nn.Dropout(drop_rate)
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layer)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layer)])

        self.w_q = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.w_k = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.w_v_x = nn.ModuleList([nn.Linear(dim, 1) for _ in range(layer)])
        self.w_v_y = nn.ModuleList([nn.Linear(dim, 1) for _ in range(layer)])
        self.w_x = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.w_y = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])

        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.fc = nn.Linear(2*dim, dim)
        self.out = nn.Linear(dim, 1)

    def softmax(self, a, mask, dim=-1):
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax

    def add(self, seq):
        norm = torch.norm(seq, dim=2, keepdim=True) #batch_size*l
        norm = F.softmax(norm, dim=1)
        output = torch.matmul(seq.transpose(1,2), norm)
        output = torch.squeeze(output, 2)
        return output
    
    def forward(self, seq1, seq2, mask1=None, mask2=None, bsz=16):
        x = F.relu(self.l1(seq1))
        y = F.relu(self.l2(seq2))
        
        for i in range(self.layer):
            x = self.norm1[i](x)
            y = self.norm2[i](y)

            q = self.w_q[i](x)
            k = self.w_k[i](y)
            d = torch.matmul(q,k.permute(0,2,1))/self.scale
            d = torch.sigmoid(d)

            v_x = F.relu(self.w_v_x[i](x))
            v_y = F.relu(self.w_v_y[i](y))
            v = torch.matmul(v_x,v_y.permute(0,2,1))
            v = v*d

            x_update = F.relu(self.w_x[i](x))
            y_update = F.relu(self.w_y[i](y))

            x = x + torch.matmul(v*mask1.view(bsz, -1, 1), y_update)
            y = y + torch.matmul(v.transpose(1,2)*mask2.view(bsz, -1, 1), x_update)

        x = self.add(x)
        y = self.add(y)

        output = torch.cat((x, y), 1)
        output = F.relu(self.fc(output))
        output = torch.sigmoid(self.out(output))

        return output

class Net(nn.Module):
    def __init__(self, geom_graph, attn_graph, in_dim=20, dim=128, n_heads=8, kernel_size=3, kernel_size_2d=(3,3), dilation=(1,1), stride=1, drop_rate=0.1, x_layer=4, y_layer=4, out_layer=2, device=device):
        super(Net, self).__init__()

        self.geom_graph_x = geom_graph(in_dim, dim, n_heads, kernel_size, kernel_size_2d, dilation, stride, drop_rate, x_layer, device)
        self.geom_graph_y = geom_graph(in_dim, dim, n_heads, kernel_size, kernel_size_2d, dilation, stride, drop_rate, y_layer, device)
        self.out_layer = attn_graph(dim, n_heads, drop_rate, out_layer, device)

    def forward(self, seq_x, seq_y, mask1, mask2, bsz):
        x = self.geom_graph_x(seq_x, mask1, bsz)
        y = self.geom_graph_y(seq_y, mask2, bsz)
        output = self.out_layer(x, y, mask1, mask2, bsz)

        return output
