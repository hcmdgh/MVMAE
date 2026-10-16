from ..common import * 


def to_canonical_etype(etype: Union[EdgeType, str],
                       etypes: Iterable[EdgeType]) -> EdgeType: 
    if isinstance(etype, tuple): 
        assert etype in etypes  
        return etype 
    elif isinstance(etype, str): 
        for _etype in etypes: 
            if etype.lower().strip() == _etype[1].lower().strip(): 
                return _etype 
            
        raise AssertionError 
    else:
        raise TypeError 
