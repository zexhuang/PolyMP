import numpy as np
import pandas as pd
import freetype as ft

from pathlib import Path
from typing import List
from tqdm import tqdm

from shapely.geometry import LinearRing, Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from shapely.affinity import rotate, skew, scale
from sklearn.model_selection import train_test_split


def _rotate(geom: BaseGeometry):
    deg = np.random.randint(-75, 75)
    return 'r', rotate(geom, deg, origin='centroid')


def _skew(geom: BaseGeometry):
    xs = np.random.randint(-45, 45)
    ys = np.random.randint(-45, 45)
    return 'sk', skew(geom, xs=xs, ys=ys, origin='centroid')


def _scale(geom: BaseGeometry):
    xs = np.random.randint(1, 20) * 0.1
    ys = np.random.randint(1, 20) * 0.1
    return 'sc', scale(geom, xfact=xs, yfact=ys, origin='centroid')


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
    contours = []   
    if isinstance(geom, Polygon):
        exter = np.asarray(geom.exterior.coords.xy)[:,0:-1] # drop_last
        contours.append(exter.shape[-1])
        
        inters = []
        for i in list(geom.interiors):
            inters.append(np.asarray(i.coords.xy)[:,0:-1]) # drop_last
            contours.append(inters[-1].shape[-1])
            
        coords = np.concatenate((exter, *inters), axis=-1) # feat_dim, num_point
    elif isinstance(geom, MultiPolygon):
        coords = []    
        for poly in geom.geoms:
            exter = np.asarray(poly.exterior.coords.xy)[:,0:-1] # drop_last
            contours.append(exter.shape[-1])
            
            inters = []
            for i in list(poly.interiors):
                inters.append(np.asarray(i.coords.xy)[:,0:-1]) # drop_last
                contours.append(inters[-1].shape[-1])
                        
            coords.append(np.concatenate((exter, *inters), axis=-1)) # feat_dim, num_point
        coords = np.concatenate(coords, axis=-1) 
    else:
        raise Exception('Wrong geom type.')
    
    contour = _contour(coords, contours)
    return coords, contour


def shortest_path(edge_index: np.array):
    import torch
    from torch_geometric.utils import to_dense_adj
    from gtda.graphs import GraphGeodesicDistance
    
    adj = to_dense_adj(torch.from_numpy(edge_index)).squeeze(0).cpu().numpy()
    adj[adj == 0] = np.Inf
    ggd = GraphGeodesicDistance(directed=False, unweighted=True).fit_transform([adj])
    return ggd.flatten()
    

def process_data(files: List[Path], 
                 letter: List[str]):
    data = []
    pbar = tqdm(total=len(files))
    for _, ttf in enumerate(files):
        face = ft.Face(str(ttf))
        face.set_char_size(32 * 32)
        font = str(ttf.parent)
        pbar.update(1)
        
        for _, l in enumerate(letter):
            face.load_char(l, ft.FT_LOAD_DEFAULT
                            | ft.FT_LOAD_NO_BITMAP)
            slot = face.glyph
            points = slot.outline.points
            contours = slot.outline.contours
            # Construct polygons with holes
            rings = []
            start = 0
            for _, end in enumerate(contours):
                if len(contours) == 1:
                    contour = points[start:]
                else:
                    contour = points[start:end + 1]
                    start = end + 1

                if len(contour) > 2:
                    rings.append(LinearRing(contour))
            # Font types that can't recognizes mult-part polygons
            if not rings: continue
            geom = Polygon(rings[0], rings[1:])      
            # Sanity check
            if not geom.is_valid:
                geom = make_valid(geom)
            # Non-empty (single- or multi-part) polygons    
            if not geom.is_empty and \
               not isinstance(geom, GeometryCollection):
                data_o = {}
                data_o['name'] = l
                data_o['font'] = font
                data_o['trans'] = 'o'
                data_o['geom'] = geom.wkt
                data_o['pos'], data_o['contour'] = to_edge_index(geom)
                data_o['ggd'] = shortest_path(data_o['contour'])
                
                data_r = {}
                data_r['name'] = l
                data_r['font'] = font
                data_r['trans'], geom_r = _rotate(geom)
                data_r['geom'] = geom_r.wkt
                data_r['pos'], data_r['contour'] = to_edge_index(geom_r)
                data_r['ggd'] = shortest_path(data_r['contour'])
                
                data_sk = {}
                data_sk['name'] = l
                data_sk['font'] = font
                data_sk['trans'], geom_sk = _skew(geom)
                data_sk['geom'] = geom_sk.wkt
                data_sk['pos'], data_sk['contour'] = to_edge_index(geom_sk)
                data_sk['ggd'] = shortest_path(data_sk['contour'])
                
                data_sc = {}
                data_sc['name'] = l
                data_sc['font'] = font
                data_sc['trans'], geom_sc = _scale(geom)
                data_sc['geom'] = geom_sc.wkt
                data_sc['pos'], data_sc['contour'] = to_edge_index(geom_sc)
                data_sc['ggd'] = shortest_path(data_sc['contour'])
                
                data += [data_o, data_r, data_sk, data_sc]
    pbar.close()
    return pd.DataFrame(data)
      
            
if __name__ == "__main__":
    glyph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']

            # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            #  'Á', 'Ä', 'Č', 'Ď', 'É', 'Í', 'Ĺ', 'Ľ', 'Ň', 'Ó', 
            #  'Ô', 'Ŕ', 'Š', 'Ť', 'Ú', 'Ý', 'Ž'] 
    
    path = Path(__file__).resolve().parent
    df = process_data(list((path / 'sans').glob('*.ttf')) \
                      + list((path / 'serif').glob('*.ttf')), glyph)
    df.to_pickle(path / 'glyph.pkl')
    # split data to train, val and test set.   
    if not (path / 'index.txt').exists():
        train, test = train_test_split(df.index, test_size=0.2)
        train, val = train_test_split(train, test_size=0.25)
        with open((path / 'index.txt'), 'w') as f:
            for idx in [train, val, test]:
                f.write(f"{' '.join(str(i) for i in idx)}\n")
    else:
        with open((path / 'index.txt'), 'r') as f:            
            lines = f.readlines()            
            train = [int(idx) for idx in lines[0].split(' ')]
            val = [int(idx) for idx in lines[1].split(' ')]
            test = [int(idx) for idx in lines[2].split(' ')]
    
    train_df, val_df, test_df = df.iloc[train], df.iloc[val], df.iloc[test]
    for f in ['train', 'val', 'test']: (path / f).mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(path / 'train' / 'glyph_train.pkl')
    val_df.to_pickle(path / 'val' / 'glyph_val.pkl')
    test_df.to_pickle(path / 'test' / 'glyph_test.pkl')
    

