pytorch basic
=============

Contents
--------
- [tensor](#tensor)

tensor
------

### Difference between torch.tensor, torch.Tensor and torch.from_numpy

1. torch.tensor infers the dtype automatically, and accept python data type and numpy array  
2. torch.Tensor returns a torch.FloatTensor  
3. torch.from_numpy automatically inherits input array dtype, only accept numpy array  

```
>>> torch.tensor([1, 2, 3]).dtype
torch.int64
>>> torch.Tensor([1, 2, 3]).dtype
torch.float32
```

```
>>> import numpy as np
>>> import torch
>>> a = [1, 2, 3]
>>> b = np.array(a)
>>> c = torch.tensor(a)
>>> d = torch.from_numpy(b)
>>> a
[1, 2, 3]
>>> b
array([1, 2, 3])
>>> c
tensor([1, 2, 3])
>>> d
tensor([1, 2, 3])
>>> b.dtype
dtype('int64')
>>> c.dtype
torch.int64
>>> d.dtype
torch.int64
>>> e = torch.Tensor(a)
>>> e.dtype
torch.float32
```
