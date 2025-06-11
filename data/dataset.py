
import torch 
import pathlib
import numpy as np
import pandas as pd

from typing import Union
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import NormalizeScale

from tqdm import tqdm
from typing import Any
from shapely import wkt
from DDSL.experiments.exp2_mnist.loader import poly2ve
from DDSL.ddsl.ddsl import *


class PolygonDataset(Dataset):
    def __init__(self, 
                 dataset: Union[str, pathlib.Path, pd.DataFrame], 
                 cls: dict,
                 freqXY = [16, 16],
                 extent = (-1,1,-1,1),
                 j = 2,
                 min_freqXY = 1, 
                 max_freqXY= 16, 
                 mid_freqXY = None,
                 smoothing = 'gaussian', 
                 fft_sigma=2.0,
                 freq_init = "fft",
                 elem_batch=100,
                 mode='density', 
                 embed_norm: Any = 'l2',
                 transform=NormalizeScale()):
        super().__init__()        
        if isinstance(dataset, str or pathlib.Path):
            df = pd.read_pickle(dataset) 
        elif isinstance(dataset, pd.DataFrame):
            df = dataset
        # Graph feature                
        self.pos = list(df['pos'])
        self.edge = list(df['contour'])
        self.trans = list(df['trans'])
        
        self.transform = transform
        
        # Label
        self.name = list(df['name'])
        self.y = torch.tensor([cls[n] for n in self.name]).float()
        # Spectral feature     
        self.spec_feature = []    
        self.geoms = df.geom.tolist()
        
        periodXY = make_periodXY(extent)
        
        ddsl_spec = DDSL_spec(res = freqXY, 
                              t = periodXY, 
                              j = j, 
                              min_freqXY = min_freqXY, 
                              max_freqXY = max_freqXY, 
                              mid_freqXY = mid_freqXY,
                              freq_init = freq_init, 
                              elem_batch = elem_batch, 
                              mode = mode)
        
        for _, geom in enumerate(tqdm(self.geoms)):
            P = wkt.loads(geom)
            V, E = poly2ve(P)
            # Rescale to (-1, -1)
            V = V - V.mean(axis=-2, keepdims=True)
            scale = (1 / np.absolute(V).max()) * 0.999999
            V *= scale
            # Random Translation
            V += 1e-6*np.random.rand(*V.shape)
            
            V = torch.tensor(V, dtype=torch.float64, requires_grad=False)
            E = torch.LongTensor(E)
            D = torch.ones(E.shape[0], 1, dtype=torch.float64)
            F = ddsl_spec(V.view(1, -1, 2), E.view(1, -1, 2), D.view(1, -1, 1))
            
            poly_nuft_embeds = F.flatten().to(V.dtype)
            poly_nuft_embeds[poly_nuft_embeds.isnan()] = 0
                
            if embed_norm == None or embed_norm == "F":
                poly_nuft_embeds_ = poly_nuft_embeds
            elif embed_norm == "l2":
                poly_nuft_embeds_norm = torch.norm(poly_nuft_embeds, p=2, dim=-1, keepdim=True)
                poly_nuft_embeds_ = torch.div(poly_nuft_embeds, poly_nuft_embeds_norm)
            
            self.spec_feature.append(poly_nuft_embeds_.view(1, -1).float())
        
    def len(self):
        return len(self.y)
    
    def get(self, idx):                               
        data = Data(pos=torch.from_numpy(self.pos[idx].T).float(),
                    edge_index=torch.from_numpy(self.edge[idx]).long(),
                    y=self.y[idx].long())
        
        data.num_nodes = data.pos.size(0)
        data.spec = self.spec_feature[idx]
        
        if self.transform:
            data = self.transform(data)
        return data