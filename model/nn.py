import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Union

from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import (GCNConv,
                                GINConv,
                                BatchNorm, 
                                LayerNorm,
                                global_add_pool,
                                global_mean_pool,
                                global_max_pool)
from torch_geometric.typing import Adj, PairTensor
from torch_geometric.utils import to_dense_batch


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 norm):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hid_channels)
        self.lin2 = nn.Linear(hid_channels, out_channels)
        self.norm = norm(hid_channels)
        self.act = nn.PReLU()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.act(self.norm(x))
        x = self.lin2(x)
        return x
        

"""VeerCNN

Paper: Deep Learning for Classiﬁcation Tasks on Geospatial Vector Polygons
Author: Veer et al.
Site: https://arxiv.org/abs/1806.03857
"""


class CNN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=32,
                      kernel_size=5),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.MaxPool1d(3, stride=3))
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=5),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.MaxPool1d(3, stride=3))        
        self.cls = MLP(in_channels=64, 
                       hid_channels=32, 
                       out_channels=out_channels, 
                       norm=nn.BatchNorm1d)

    def forward(self, data, max_num_nodes=1024):
        x = data.pos
        batch = data.batch
        # Node to dense batch repr
        x_list = []    
        batch_size = batch.max() + 1
        ids = torch.arange(0, batch_size)
        for id in ids:
            x_batch = x[batch == id]
            while x_batch.size(0) > max_num_nodes:
                prob = torch.rand(x_batch.size(0), device=x_batch.device)
                mask = prob > 0.1
                x_batch = x_batch[mask]

            x_dense = to_dense_batch(x_batch, 
                                     fill_value=0.0, 
                                     max_num_nodes=max_num_nodes)[0].permute(0, 2, 1)
            x_list.append(x_dense)
            
        x = torch.cat(x_list, dim=0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        return self.cls(x)
        
    
"""Deepsets

Paper: Deep Sets
Author: Zaheer et al.
Site: https://github.com/manzilzaheer/DeepSets
"""


class Phi(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 aggregation_fn='max'):
        super().__init__()
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn
        self.Gamma = nn.Linear(in_channels, out_channels)
        self.Lambda = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = LayerNorm(out_channels)
        self.act = nn.PReLU()

    def forward(self, x, batch):        
        # Apply aggregation over point set
        from torch_scatter import scatter
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm[batch, :]
        x = self.norm(x)
        return self.act(x)


class Deepset(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels):
        super(Deepset, self).__init__()
        self.conv1 = Phi(in_channels, hid_channels)         # Equivariant Layer
        self.conv2 = Phi(hid_channels, hid_channels)        # Equivariant Layer        
        self.cls = MLP(hid_channels, 
                       hid_channels//2, 
                       out_channels, 
                       norm=LayerNorm)                  
        
    def forward(self, data):
        x = data.pos
        batch = data.batch
        
        x = self.conv1(x, batch)
        x = self.conv2(x, batch)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]  
        return self.cls(x)


"""Set Transformer

Paper: Set Transformer:
       A Framework for Attention-based Permutation-Invariant Neural Networks
Author: Lee et al.
Site: https://github.com/juho-lee/set_transformer
"""


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.BatchNorm1d(dim_V)
            self.ln1 = nn.BatchNorm1d(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(
            Q_.bmm(
                K_.transpose(
                    1,
                    2)) /
            math.sqrt(
                self.dim_V),
            2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(
            O.permute(0, -1, 1)).permute(0, -1, 1)
        O = O + F.leaky_relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(
            O.permute(0, -1, 1)).permute(0, -1, 1)
        return O


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, 
                 dim_input, 
                 dim_hidden, 
                 dim_output,
                 num_seeds=1, num_heads=4, 
                 num_inds=32, ln=True):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_seeds, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln))        
        self.cls = MLP(dim_hidden, 
                       dim_hidden//2, 
                       dim_output, 
                       norm=nn.BatchNorm1d)  

    def forward(self, data, max_num_nodes=1024):
        x = data.pos
        batch = data.batch
        # Node to dense batch repr
        x_list = []    
        batch_size = batch.max() + 1
        ids = torch.arange(0, batch_size)
        for id in ids:
            x_batch = x[batch == id]
            while x_batch.size(0) > max_num_nodes:
                prob = torch.rand(x_batch.size(0), device=x_batch.device)
                mask = prob > 0.1
                x_batch = x_batch[mask]

            x_dense = to_dense_batch(x_batch, 
                                     fill_value=0.0, 
                                     max_num_nodes=max_num_nodes)[0]
            x_list.append(x_dense)
            
        x = torch.cat(x_list, dim=0)
        x = self.enc(x)
        x = self.dec(x).squeeze(dim=1)
        x = self.cls(x)
        return x 


"""GCAE

Paper: Graph convolutional autoencoder model for the shape coding and cognition of buildings in maps
Author: Yan et al.
Site: https://www.tandfonline.com/doi/abs/10.1080/13658816.2020.1768260?journalCode=tgis20
"""


class GCNBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels):
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = LayerNorm(out_channels)
        self.act = nn.PReLU()
        
    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return self.act(x)
    
    
class GCN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hid_channels, 
                 out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNBlock(in_channels, hid_channels)
        self.conv2 = GCNBlock(hid_channels, hid_channels)
        self.cls = MLP(hid_channels, 
                       hid_channels//2, 
                       out_channels, 
                       norm=LayerNorm)  
        
    def forward(self, data):
        x = data.pos
        batch = data.batch
        edge_index = data.edge_index
        
        x = self.conv1(x, edge_index, batch)
        x = self.conv2(x, edge_index, batch)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]  
        return self.cls(x)
    
    
"""NUFTSpec

Paper: Towards general-purpose representation learning of polygonal geometries
Author: Mai et al.
Site: https://github.com/gengchenmai/polygon_encoder
"""

class NUFTSpecMLP(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hid_channels, 
                 out_channels):
        super(NUFTSpecMLP, self).__init__()
        self.enc = nn.Sequential(nn.Linear(in_channels, hid_channels * 2),
                                 nn.LayerNorm(hid_channels * 2),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hid_channels * 2, hid_channels),
                                 nn.LayerNorm(hid_channels),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))  
        self.cls = MLP(hid_channels, 
                       hid_channels//2, 
                       out_channels, 
                       norm=LayerNorm)  
        
    def forward(self, data):
        x = data.spec
        x = self.enc(x)
        return self.cls(x)
    

"""DSC-NMP

Paper: Propagation Enhanced Neural Message Passing for Graph Representation Learning
Author: Fan et al.
Site: https://github.com/xiaolongo/SelfConNMP/blob/main/dsc_nmp.py
"""
    
    
class DSCNMP(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hid_channels, 
                 out_channels):
        super(DSCNMP, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hid_channels), 
                            nn.ReLU(),
                            nn.Linear(hid_channels, hid_channels),
                            nn.ReLU(),
                            nn.BatchNorm1d(hid_channels))    
        self.conv1 = GINConv(nn1)
        
        nn2 = nn.Sequential(nn.Linear(hid_channels, hid_channels), 
                            nn.ReLU(),
                            nn.Linear(hid_channels, hid_channels),
                            nn.ReLU(),
                            nn.BatchNorm1d(hid_channels))        
        self.conv2 = GINConv(nn2)

        self.fc1 = nn.Sequential(nn.Linear(in_channels, hid_channels), 
                                 nn.ReLU(),
                                 nn.BatchNorm1d(hid_channels))
        self.fc2 = nn.Sequential(nn.Linear(hid_channels, hid_channels), 
                                 nn.ReLU(),
                                 nn.BatchNorm1d(hid_channels))

        self.cls = MLP(hid_channels, 
                       hid_channels//2, 
                       out_channels, 
                       norm=BatchNorm)  

        
    def forward(self, data):
        x, batch, edge_index = data.pos, data.batch, data.edge_index
        
        x0_g = global_add_pool(x, batch)
        x0_g = self.fc1(x0_g)

        x1 = self.conv1(x, edge_index)
        x1_g = global_add_pool(x1, batch)
        x1_g = self.fc2(x0_g + x1_g)

        x2 = self.conv2(x1, edge_index)
        x2_g = global_add_pool(x2, batch)
        x2_g = self.fc2(x0_g + x1_g + x2_g)
        
        x = F.dropout(x2_g, p=0.5, training=self.training)
        return self.cls(x)
        
    
"""EdgeConv

Paper: Dynamic Graph CNN for Learning on Point Clouds
Author: Wang et al.
Site: https://github.com/WangYueFt/dgcnn
"""


class EdgeConv(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        
        self.affine_w = Parameter(torch.ones(nn.in_features // 2))
        self.affine_b = Parameter(torch.zeros(nn.in_features // 2))

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        std_x = torch.std(x_j - x_i)
        x_rel = (x_j - x_i) / (std_x + 1e-5)
        x_rel = self.affine_w * x_rel + self.affine_b

        return self.nn(torch.cat([x_i, x_rel], dim=-1))


class EdgeConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels):
        super(EdgeConvBlock, self).__init__()
        self.conv = EdgeConv(nn.Linear(in_channels*2, out_channels))
        self.norm = LayerNorm(out_channels)
        self.act = nn.PReLU()
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return self.act(x)
    
    
class PolyMP(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hid_channels, 
                 out_channels):
        super(PolyMP, self).__init__()
        self.mp1 = EdgeConvBlock(in_channels, hid_channels)
        self.mp2 = EdgeConvBlock(hid_channels, hid_channels)
        
        self.cls = MLP(hid_channels, 
                       hid_channels//2, 
                       out_channels, 
                       norm=LayerNorm)  
        
    def forward(self, data):
        x, batch, edge_index = data.pos, data.batch, data.edge_index
        x = self.mp1(x, edge_index)
        x = self.mp2(x, edge_index)
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]  
        return self.cls(x)
    
    
class DSCPolyMP(nn.Module):
    """
    Densely Self-Connected Neural Message Passing mechanism from the "Propagation Enhanced Neural Message Passing
    for Graph Representation Learning" paper by Fan et al..
    """
    
    def __init__(self, 
                 in_channels, 
                 hid_channels, 
                 out_channels):
        super(DSCPolyMP, self).__init__()
        self.mp1 = EdgeConvBlock(in_channels, hid_channels)
        self.mp2 = EdgeConvBlock(hid_channels, hid_channels)
        
        self.fc1 = nn.Sequential(nn.Linear(in_channels, hid_channels), 
                                 nn.LayerNorm(hid_channels),
                                 nn.PReLU())
        self.fc2 = nn.Sequential(nn.Linear(hid_channels, hid_channels), 
                                 nn.LayerNorm(hid_channels),
                                 nn.PReLU())
        
        self.cls = MLP(hid_channels, 
                       hid_channels//2, 
                       out_channels, 
                       norm=LayerNorm)  
        
    def forward(self, data):
        x, batch, edge_index = data.pos, data.batch, data.edge_index
        
        x0_g = global_max_pool(x, batch)   # [batch_size, hidden_channels]  
        x0_g = self.fc1(x0_g)
        
        x1 = self.mp1(x, edge_index)
        x1_g = global_max_pool(x1, batch)
        x1_g = self.fc2(x0_g + x1_g)
        
        x2 = self.mp2(x1, edge_index)
        x2_g = global_max_pool(x2, batch)
        x2_g = self.fc2(x0_g + x1_g + x2_g)
        
        x = F.dropout(x2_g, p=0.25, training=self.training)
        return self.cls(x)
    
    
import yaml
from pathlib import Path
from typing import Union


def build_model(cfg: Union[str, Path, dict]):
    if isinstance(cfg, str or Path):
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)    
            
    if cfg['nn'] == 'cnn':
        return CNN(cfg['in_channels'], 
                   cfg['out_channels'])
    elif cfg['nn'] == 'deepset':
        return Deepset(cfg['in_channels'], 
                       cfg['hid_channels'], 
                       cfg['out_channels'])
    elif cfg['nn'] == 'gcn':
        return GCN(cfg['in_channels'], 
                   cfg['hid_channels'], 
                   cfg['out_channels'])
    elif cfg['nn'] == 'transformer':
        return SetTransformer(cfg['in_channels'], 
                              cfg['hid_channels'], 
                              cfg['out_channels'])
    elif cfg['nn'] == 'nuft_spec_mlp':
        return NUFTSpecMLP(288, 
                           cfg['hid_channels'], 
                           cfg['out_channels'])
    elif cfg['nn'] == 'dsc_nmp':
        return DSCNMP(cfg['in_channels'], 
                      cfg['hid_channels'], 
                      cfg['out_channels'])
    elif cfg['nn'] == 'polymp':
        return PolyMP(cfg['in_channels'], 
                      cfg['hid_channels'], 
                      cfg['out_channels'])
    elif cfg['nn'] == 'dsc_polymp':
        return DSCPolyMP(cfg['in_channels'], 
                         cfg['hid_channels'], 
                         cfg['out_channels'])
        
        
def finetune_model(cfg: Union[str, Path, dict]):
    if isinstance(cfg, str or Path):
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)    
            
    if cfg['nn'] == 'cnn':
        model = CNN(cfg['in_channels'], 
                    cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels'] // 2, 
                        cfg['out_channels'],
                        norm=nn.BatchNorm1d)  
        
    elif cfg['nn'] == 'deepset':
        model = Deepset(cfg['in_channels'], 
                        cfg['hid_channels'], 
                        cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels'] // 2, 
                        cfg['out_channels'],
                        norm=LayerNorm)  
        
    elif cfg['nn'] == 'gcn':
        model = GCN(cfg['in_channels'], 
                    cfg['hid_channels'], 
                    cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels'] // 2, 
                        cfg['out_channels'],
                        norm=LayerNorm)  
        
    elif cfg['nn'] == 'transformer':
        model = SetTransformer(cfg['in_channels'], 
                               cfg['hid_channels'], 
                               cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels'] // 2, 
                        cfg['out_channels'],
                        norm=nn.BatchNorm1d)  
        
    elif cfg['nn'] == 'nuft_spec_mlp':
        model = NUFTSpecMLP(288, 
                            cfg['hid_channels'], 
                            cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels']//2, 
                        cfg['out_channels'], 
                        norm=LayerNorm)
        
    elif cfg['nn'] == 'dsc_nmp':
        model = DSCNMP(cfg['in_channels'], 
                       cfg['hid_channels'], 
                       cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels'] // 2, 
                        cfg['out_channels'],
                        norm=BatchNorm)  
        
    elif cfg['nn'] == 'polymp':
        model = PolyMP(cfg['in_channels'], 
                       cfg['hid_channels'], 
                       cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels'] // 2, 
                        cfg['out_channels'],
                        norm=LayerNorm)  
        
    elif cfg['nn'] == 'dsc_polymp':
        model = DSCPolyMP(cfg['in_channels'], 
                          cfg['hid_channels'], 
                          cfg['out_channels'])
        
        model.load_state_dict(torch.load(f"{cfg['path']}/ckpt/{cfg['ckpt']}")['params'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.cls = MLP(cfg['hid_channels'], 
                        cfg['hid_channels'] // 2, 
                        cfg['out_channels'], 
                        norm=LayerNorm)  
    return model
    

