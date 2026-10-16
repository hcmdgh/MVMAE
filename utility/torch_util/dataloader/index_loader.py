from ..common import * 


class IndexLoader:
    def __init__(self,
                 indices: Any,
                 batch_size: int,
                 shuffle: bool,
                 collate_func: Optional[Callable] = None, 
                 drop_last: bool = False):
        if isinstance(indices, Tensor):
            indices = indices.detach().cpu().numpy() 
        
        if isinstance(indices, int):
            self.indices = np.arange(indices) 
        elif isinstance(indices, np.ndarray):
            if indices.dtype == np.int64:
                assert indices.ndim == 1  
                self.indices = indices 
            elif indices.dtype == bool: 
                assert indices.ndim == 1  
                self.indices = indices.nonzero()[0]
            else:
                raise TypeError 
        else:
            raise TypeError 
        
        assert isinstance(self.indices, ndarray) and self.indices.dtype == np.int64 and self.indices.ndim == 1  
        
        if batch_size == -1: 
            batch_size = int(1e9) 
        
        self.batch_size = batch_size          
        self.shuffle = shuffle 
        self.drop_last = drop_last
        self.collate_func = collate_func
        
    def __iter__(self) -> Iterator:
        N = len(self.indices)
        
        if self.shuffle:
            perm = np.random.permutation(N)
        else:
            perm = np.arange(N)
            
        for i in range(0, N, self.batch_size):
            batch = self.indices[perm[i: i + self.batch_size]]
            
            if self.drop_last and len(batch) < self.batch_size:
                break 
            
            if self.collate_func is None:
                yield batch 
            else:
                yield self.collate_func(batch)

    def __len__(self) -> int: 
        N = len(self.indices)
        
        if self.drop_last:
            _len = N // self.batch_size 
        else:
            _len = math.ceil(N / self.batch_size) 

        return _len 
