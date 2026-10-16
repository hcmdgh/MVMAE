import os 
from torch_geometric.datasets import TUDataset as _TUDataset

from ...common import * 


class TUDataset:
    def __init__(self,
                 name: str):
        self.dataset = _TUDataset(
            root = os.path.expanduser('~/dataset/PyG/TUDataset/raw'), 
            name = name,  
        )         
