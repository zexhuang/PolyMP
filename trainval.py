import yaml

from torch_geometric.loader import DataLoader
from data.dataset import PolygonDataset
from train.trainer import Trainer
from model.nn import build_model


if __name__ == '__main__':    
    with open('cfg/train_glyph.yaml', 'r') as f:
        cfg = yaml.safe_load(f)    
        cls = [k for k, _ in cfg['cls'].items()]
        
        for frac in [0.0, 0.2, 0.4, 0.6, 0.8]:
            cfg['frac'] = frac
        
            import pandas as pd
            train_df = pd.read_pickle(cfg['train'])
            train_df_o = train_df[(train_df.name.isin(cls)) & (train_df.trans=='o')]            
            train_df_trans = train_df[(train_df.name.isin(cls)) & ~(train_df.trans=='o')]
            train_df_trans = train_df_trans.sample(n=int(len(train_df_o)*cfg['frac']), random_state=123, ignore_index=True)
            train_df_o = train_df_o.sample(n=int(len(train_df_o)*(1-cfg['frac'])), random_state=123, ignore_index=True)   
            train_df = pd.concat([train_df_o, train_df_trans], axis=0, ignore_index=True)
            train_loader = DataLoader(PolygonDataset(train_df, cfg['cls']), 
                                      batch_size=cfg['batch'],
                                      num_workers=cfg['worker'])
            
            val_df = pd.read_pickle(cfg['val'])
            val_df_o = val_df[(val_df.name.isin(cls)) & (val_df.trans=='o')]            
            val_df_trans = val_df[(val_df.name.isin(cls)) & ~(val_df.trans=='o')]
            val_df_trans = val_df_trans.sample(n=int(len(val_df_o)*cfg['frac']), random_state=123, ignore_index=True)
            val_df_o = val_df_o.sample(n=int(len(val_df_o)*(1-cfg['frac'])), random_state=123, ignore_index=True)
            val_df = pd.concat([val_df_o, val_df_trans], axis=0, ignore_index=True)
            val_loader = DataLoader(PolygonDataset(val_df, cfg['cls']), 
                                    batch_size=cfg['batch'],
                                    num_workers=cfg['worker'])
            
            for nn in cfg['model_list']:
                cfg['nn'] = nn
                cfg['path'] = f"save/frac{cfg['frac']}/{nn}"
                model = build_model(cfg=cfg)
                trainer = Trainer(cfg=cfg) 
                trainer.fit(model, 
                            train_loader=train_loader, 
                            val_loader=val_loader)
