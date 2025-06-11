import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from typing import List
from tqdm import tqdm

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from sklearn.model_selection import train_test_split


def _contour(coords: np.ndarray, 
             contours: List[int]):
    node_idx = np.arange(0, coords.shape[1], dtype=np.int64)
    src_list, tgt_list = [], []
    
    start = 0
    for idx, _ in enumerate(contours):
        end = sum(contours[:idx+1])
        
        ids = node_idx[start:end]
        src = np.concatenate([ids, ids], axis=0)  
        tgt = np.concatenate([np.roll(ids, shift=-1, axis=0), 
                              np.roll(ids, shift=1, axis=0)], axis=0)
        src_list.append(src)
        tgt_list.append(tgt)
        start = end
        
    src = np.concatenate(src_list)
    tgt = np.concatenate(tgt_list)
    edge_index = np.concatenate([src.reshape(1, -1), tgt.reshape(1, -1)], axis=0)
    return edge_index 


def to_edge_index(geom: BaseGeometry):
    assert isinstance(geom, Polygon)
    
    contours = []   
    exter = np.asarray(geom.exterior.coords.xy)[:,0:-1] # drop_last
    contours.append(exter.shape[-1])
    
    inters = []
    for i in list(geom.interiors):
        inters.append(np.asarray(i.coords.xy)[:,0:-1]) # drop_last
        contours.append(inters[-1].shape[-1])
        
    coords = np.concatenate((exter, *inters), axis=-1) # feat_dim, num_point
    edge_index = _contour(coords, contours)
    return coords, edge_index


def shortest_path(edge_index: np.array):
    import torch
    from torch_geometric.utils import to_dense_adj
    from gtda.graphs import GraphGeodesicDistance
    
    adj = to_dense_adj(torch.from_numpy(edge_index)).squeeze(0).cpu().numpy()
    adj[adj == 0] = np.Inf
    ggd = GraphGeodesicDistance(directed=False, unweighted=True).fit_transform([adj])
    return ggd.flatten()


def process_data(gdf: gpd.GeoDataFrame):
    data = []
    pbar = tqdm(total=len(gdf.index))
    for _, row in gdf.iterrows():
        sample = {}
        geom = row.geometry
                
        if not geom.is_valid:
            geom = make_valid(geom)
        
        sample['name'] = row['name']
        sample['trans'] = 'o' if pd.isna(row.valid) else 'r'
        sample['geom'] = geom.wkt
        sample['pos'], sample['contour'] = to_edge_index(geom)
        sample['ggd'] = shortest_path(sample['contour'])

        data.append(sample)
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(data)
    
    
if __name__ == "__main__":
    path = Path(__file__).resolve().parent
    df = process_data(gdf = pd.concat([gpd.read_file(path / 'data' / 'test_5010r.shp'),  # OSM 
                                       gpd.read_file(path / 'data' / 'test_5010.shp')])) # Rotation & Reflection
    
    df.to_pickle(path / 'osm.pkl')
    
    # split data to train, val and test set.   
    if not (path / 'index.txt').exists():
        train, test = train_test_split(df.index, test_size=0.5)
        with open((path / 'index.txt'), 'w') as f:
            for idx in [train, test]:
                f.write(f"{' '.join(str(i) for i in idx)}\n")
    else:
        with open((path / 'index.txt'), 'r') as f:            
            lines = f.readlines()            
            train = [int(idx) for idx in lines[0].split(' ')]
            test = [int(idx) for idx in lines[1].split(' ')]
            
    train_df, test_df = df.iloc[train], df.iloc[test]
    for f in ['train', 'test']: (path / f).mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(path / 'train' / 'osm_train.pkl')
    test_df.to_pickle(path / 'test' / 'osm_test.pkl')