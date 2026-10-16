# PyTorch、NumPy系列
import torch 
from torch import Tensor 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.multiprocessing as mp 
from torch.nn.utils import clip_grad_norm_ 
import numpy as np 
from numpy import ndarray 
import scipy 
import scipy.sparse as sp 
from typing import Union 

IntTensor = FloatTensor = BoolTensor = FloatScalarTensor = SparseTensor = Tensor 
IntArray = FloatArray = BoolArray = ndarray 
FloatArrayTensor = IntArrayTensor = Union[Tensor, ndarray]


# DGL、PyG系列
import torch_scatter 
import torch_sparse 
import dgl 
import dgl.sparse as dglsp 

EdgeIndex = IntTensor 
NodeType = str 
EdgeType = tuple[str, str, str]


# Python标准库
import random 
import math 
import os
import sys 
import time
import shutil 
import uuid 
from datetime import datetime, date  
from dataclasses import dataclass, asdict
from collections import defaultdict, namedtuple, Counter 
import itertools
import functools 
import argparse 
import lzma 
import traceback 
import json 
import csv 
import copy 
import pickle 
from pprint import pprint, pformat
from typing import Optional, Any, Union, Callable, Literal, Iterable, Iterator

INF = int(1e9)
pprint = functools.partial(pprint, sort_dicts=False)
pformat = functools.partial(pformat, sort_dicts=False)


# 常用第三方库
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm 
import wandb 
