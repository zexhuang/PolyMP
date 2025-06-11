import yaml

from torch_geometric.loader import DataLoader
from data.dataset import PolygonDataset
from train.trainer import Trainer
from model.nn import finetune_model


if __name__ == '__main__':    
    with open('cfg/finetune_osm.yaml', 'r') as f:
        cfg = yaml.safe_load(f)    
        
        cls = [k for k, _ in cfg['cls'].items()]
        
    import pandas as pd
    train_df = pd.read_pickle(cfg['osm_train'])
    train_df = train_df[train_df.name.isin(cls)]
    
    val_df = pd.read_pickle(cfg['osm_test'])
    val_df = val_df[val_df.name.isin(cls)]   
    
    if cfg['normalised']:
        train_loader = DataLoader(PolygonDataset(train_df, cfg['cls']), 
                                  batch_size=cfg['batch'],
                                  num_workers=cfg['worker'])
                
        val_loader = DataLoader(PolygonDataset(val_df, cfg['cls']), 
                                batch_size=cfg['batch'],
                                num_workers=cfg['worker'])
        
        for nn in cfg['model_list']:
            cfg['nn'] = nn
            cfg['path'] = f"save/finetune/normalised/{nn}"
            model = finetune_model(cfg=cfg)
            trainer = Trainer(cfg=cfg) 
            trainer.fit(model, 
                        train_loader=train_loader, 
                        val_loader=val_loader)        
    if not cfg['normalised']:
        train_loader = DataLoader(PolygonDataset(train_df, cfg['cls'], transform=None), 
                                  batch_size=cfg['batch'],
                                  num_workers=cfg['worker'])
                
        val_loader = DataLoader(PolygonDataset(val_df, cfg['cls'], transform=None), 
                                batch_size=cfg['batch'],
                                num_workers=cfg['worker'])
        
        for nn in cfg['model_list']:
            cfg['nn'] = nn
            cfg['path'] = f"save/finetune/unnormalised/{nn}" 
            model = finetune_model(cfg=cfg)
            trainer = Trainer(cfg=cfg) 
            trainer.fit(model, 
                        train_loader=train_loader, 
                        val_loader=val_loader)