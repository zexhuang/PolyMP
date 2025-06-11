import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="dark", 
              palette="pastel")
import numpy as np


def draw_graph(pos: np.ndarray, 
               edge: np.ndarray = None, 
               mask=None, 
               ax=None):      
    # Subplots
    if ax == None: ax = plt.subplot(1, 1, 1)  
    # Edges
    if edge is not None:
        x = [pos[:,0][edge[:,0]], pos[:,0][edge[:,1]]] # x_src, x_tgt
        y = [pos[:,1][edge[:,0]], pos[:,1][edge[:,1]]] # y_src, y_tgt
        ax.plot(x, y, 'b-')         
    # Nodes
    mask = np.ones(pos.shape[0]) if mask is None else mask
    sns.scatterplot(x=pos[:,0], 
                    y=pos[:,1], 
                    hue=mask, 
                    palette="YlOrBr", 
                    markers='o', 
                    legend=False, 
                    s=60, ax=ax)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    return ax
