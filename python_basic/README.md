python basic
============

Contents
--------
- [variable scope](#variable-scope)
- [class attribute](#class-attribute)
- [decorator](#decorator)
- [shuffle array](#shuffle-array)

variable scope
--------------

global variables could be read in a function

```
>>> a = 10
>>> def read_var():
...     print(a)
...
>>> read_var()
10
>>>
```

change the global variable in a function, use 'global'

```
>>> a = 10
>>> def write_var():
...     global a
...     a = 20
...
>>> write_var()
>>> print(a)
20
>>>
```

class attribute
---------------

class attribute: shared by all instances

```
>>> class Student:
...     count = 0
...     def __init__(self, name):
...         self.name = name
...         Student.count += 1
...
>>> Student.count
0
>>> a = Student('a')
>>> Student.count
1
>>>
```

instance attribute: belong to instance self

```
>>> class Student:
...     name='Student'
...
>>> s = Student()
>>> s.name # no instance attribute, here is class attribute
'Student'
>>> Student.name # class attribute
'Student'
>>> s.name = 'John'
>>> s.name # priority of instance attribute higher than priority of class attribute
'John'
>>> Student.name
'Student'
>>> del s.name # delete instance attribute
>>> s.name
'Student'
>>>
```

decorator
---------

Add some feature to function without change it!!

For instance, a timer decorator

```
from time import time
from functools import wraps

def log(func):
    @wraps(func)
    def wrapper(*args, **kw):
        t = time()
        result = func(*args, **kw)
        print(f"{func.__name__}, {time()-t:.4f}")
        return result
    return wrapper

@log
def some_func():
    # do something
```

shuffle array
-------------

To shuffle two arrays in same order,  
generate shuffled index to order two arrays

```
>>> a = np.arange(10)
>>> b = a[::-1]
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b
array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
>>> shuffle_idx = np.arange(a.shape[0])
>>> np.random.shuffle(shuffle_idx)
>>> shuffle_idx
array([4, 2, 6, 0, 1, 9, 7, 8, 5, 3])
>>> a[shuffle_idx]
array([4, 2, 6, 0, 1, 9, 7, 8, 5, 3])
>>> b[shuffle_idx]
array([5, 7, 3, 9, 8, 0, 2, 1, 4, 6])
>>>

```
